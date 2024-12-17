"""Definition of the FOLDT inductor.
"""
# pylint:skip-file

from collections import defaultdict
from graphviz import Digraph
from pddlgym.inference import ProofSearchTree, CommitGoalError
from pddlgym.structs import Literal, NULLTYPE
from termcolor import colored

import heapq as hq
import itertools
import os
import numpy as np
import time


MAX_ASSIGNMENT_COUNT = 20
PRINT_MODE = "conditional"
DEBUG = False


class LearningFailure(Exception):
    pass


class DTNode:
    """
    A node in a decision tree.
    Parameters
    ----------
    idxs : [ int ] or None
        Training data that is shepherded to this node.
    parent : Node or None
        None if root. Sometimes updated after init.
    is_right_child : bool
        Whether this node is the right child of its parent.
    committed : bool
        Whether this node is definitely part of the
        decision tree, or is rather just a candidate.
    """
    id_counter = itertools.count()

    def __init__(self, idxs, parent, is_right_child, committed):
        self.idxs = idxs
        self.parent = parent
        self.is_right_child = is_right_child
        self.committed = committed

        self._id = next(self.id_counter)
        self._hash = hash(self._id)

    def __hash__(self):
        return self._hash

    def __str__(self):
        s = self.print_tree(self.parent, self.depth-1, leaf_to_root=True)
        if self.is_right_child:
            s += self.print_tree(self.parent.left, self.depth,
                                 highlight_node=self)
        s += self.print_tree(self, self.depth, highlight_node=self)
        if not self.is_right_child and self.parent is not None:
            s += self.print_tree(self.parent.right, self.depth,
                                 highlight_node=self)
        return s

    @property
    def depth(self):
        return 0 if self.parent is None else 1 + self.parent.depth

    @property
    def root(self):
        if self.parent is None:
            return self
        return self.parent.root

    @property
    def positive_ancestor(self):
        """The deepest ancestor that holds, if this node is reached."""
        if self.parent is None:
            positive_ancestor = None
        elif self.is_right_child:
            positive_ancestor = self.parent
        else:
            positive_ancestor = self.parent.positive_ancestor
        return  positive_ancestor

    @property
    def full_feature_path(self):
        """All of the features that are in the nodes here and above."""
        if self.parent is None:
            full_feature_path = [self.label]
        else:
            full_feature_path = [self.label] + self.parent.full_feature_path
        return full_feature_path

    @property
    def positive_feature_path(self):
        """Features that hold, if this node is reached."""
        if self.label is None:
            my_feature = []
        else:
            my_feature = [self.label]
        if self.positive_ancestor is None:
            positive_feature_path = my_feature
        else:
            positive_feature_path = my_feature + \
                self.positive_ancestor.positive_feature_path
        return positive_feature_path

    @property
    def negative_feature_path(self):
        """Features that are negated, if this node is reached."""
        my_feature = []
        if self.parent is None:
            return my_feature
        if self.is_right_child:
            return self.parent.negative_feature_path
        return [self.parent.get_naf_label()] + self.parent.negative_feature_path

    def get_naf_label(self):
        return self.label.negate_as_failure()

    @classmethod
    def print_tree(cls, node, depth=0, highlight_node=None, leaf_to_root=False):
        if node is None:
            return ''
        if not node.is_leaf:
            s = '%s[%s]\n' % ((depth*' ', node.feature))

            if node is highlight_node:
                s = colored(s, 'grey', attrs=['bold'])
            if not node.committed:
                s = colored(s, 'grey', attrs=['underline'])

            if leaf_to_root:
                return cls.print_tree(node.parent, depth-1,
                    highlight_node=highlight_node, leaf_to_root=leaf_to_root) + s

            return s + \
                cls.print_tree(node.left, depth+1,
                    highlight_node=highlight_node, leaf_to_root=leaf_to_root) + \
                cls.print_tree(node.right, depth+1,
                    highlight_node=highlight_node, leaf_to_root=leaf_to_root)

        s = '%s[%s]\n' % ((depth*' ', node.leaf_class))
        if node is highlight_node:
            s = colored(s, 'grey', attrs=['bold'])
        if not node.committed:
            s = colored(s, 'grey', attrs=['underline'])
        return s



class DTFeatureNode(DTNode):
    """
    An internal node in the decision tree, one with a feature
    and left/right children.
    Parameters
    ----------
    feature : Any
        The feature.
    left : DTNode
        The left child.
    right : DTNode
        The right child.
    idxs : [ int ] or None
        Training data that is shepherded to this node.
    parent : Node or None
        None if root. Sometimes updated after init.
    is_right_child : bool
        Whether this node is the right child of its parent.
    committed : bool
        Whether this node is definitely part of the
        decision tree, or is rather just a candidate.
    """
    def __init__(self, feature, left, right, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature = feature
        self.left = left
        self.right = right

        if DEBUG:
            assert isinstance(self.left, DTNode)
            assert isinstance(self.right, DTNode)
            assert self.right.is_right_child
            assert not self.left.is_right_child
            assert set(self.left.idxs) | set(self.right.idxs) == set(self.idxs)
            assert len(self.left.idxs) + len(self.right.idxs) == len(self.idxs)

    @property
    def score(self):
        """The score is the weighted-averaged loss of all leaf nodes."""
        left_size = len(self.left.idxs)
        left_score = self.left.score

        right_size = len(self.right.idxs)
        right_score = self.right.score

        left_frac = left_size / (left_size + right_size)
        right_frac = 1. - left_frac
        return left_score * left_frac + right_score * right_frac

    @property
    def descendents(self):
        """All nodes below."""
        return [self.left] + self.left.descendents + \
            [self.right] + self.right.descendents

    @property
    def leaf_nodes(self):
        return [node for node in self.descendents if node.is_leaf]

    @property
    def label(self):
        return self.feature

    @property
    def is_leaf(self):
        return False

    def copy(self):
        """TODO: make this obsolete"""
        return self.__class__(self.feature, self.left.copy(), self.right.copy(),
            [idx for idx in self.idxs], self.parent, self.is_right_child, self.committed)


class DTLeafNode(DTNode):
    """
    An leaf node in the decision tree, one with a class and no
    children.
    Parameters
    ----------
    leaf_class : Any
        The class at the leaf.
    score : float
        The classification loss for this leaf.
    idxs : [ int ] or None
        Training data that is shepherded to this node.
    parent : Node or None
        None if root. Sometimes updated after init.
    is_right_child : bool
        Whether this node is the right child of its parent.
    committed : bool
        Whether this node is definitely part of the
        decision tree, or is rather just a candidate.
    """
    FeatureNode = DTFeatureNode

    def __init__(self, leaf_class, score, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.leaf_class = leaf_class
        self.score = score

    @property
    def descendents(self):
        """All nodes below."""
        return []

    @property
    def label(self):
        return self.leaf_class

    @property
    def is_leaf(self):
        return True

    def split(self, feature, left_class, right_class,
              left_score, right_score, left_idxs, right_idxs):
        """Create a new feature node to replace this leaf."""
        if DEBUG:
            assert set(self.idxs) == set(left_idxs) | set(right_idxs)
            assert len(self.idxs) == len(left_idxs) + len(right_idxs)

        LeafNode = self.__class__
        left_leaf = LeafNode(left_class, left_score, left_idxs,
                             None, False, False)
        right_leaf = LeafNode(right_class, right_score, right_idxs,
                              None, True, False)
        new_node = self.FeatureNode(feature, left_leaf, right_leaf,
                                    self.idxs, self.parent, self.is_right_child,
                                    self.committed)
        left_leaf.parent = new_node
        right_leaf.parent = new_node
        return new_node

    def copy(self):
        """TODO: make this obsolete"""
        return self.__class__(self.leaf_class, self.score,
                              [idx for idx in self.idxs], self.parent,
                              self.is_right_child, self.committed)



class DecisionTreeClassifier:
    """
    A standard decision tree classifier, except:
      -Features are assumed to be binary
      -Some features may be present in certain inputs but not others
    Usage:
    model = DecisionTreeClassifier()
    model.fit(X, Y)
    model.predict(test_X)
    Parameters
    ----------
    max_depth : int or np.inf
        The maximum depth that will be considered during learning.
    root_feature : Any
        If given, this feature will be used at the root.
    seed : int
        A random seed that is used to stochastically order features.
    criterion : str
        What decision tree criterion to learn with.
    randomize_feature_order : bool
        If True, use the seed to randomize feature order.
    """
    LeafNode = DTLeafNode

    def __init__(self, root_feature=None, max_depth=np.inf, seed=0,
                 criterion='classification', randomize_feature_order=True,
                 allow_commit_exception=True, max_feature_length=np.inf,
                 max_learning_time=np.inf, max_exceeded_strategy='fail',
                 greedy_improvement_threshold=1e-5, bag_predicates=False,
                 bag_features=False):
        self.is_fit = False
        self.max_depth = max_depth
        self.root_feature = root_feature
        self.random = np.random.RandomState(seed)
        self.criterion = criterion
        self.randomize_feature_order = randomize_feature_order
        self.allow_commit_exception = allow_commit_exception
        self.max_feature_length = max_feature_length
        self.max_learning_time = max_learning_time
        self.max_exceeded_strategy = max_exceeded_strategy
        self.greedy_improvement_threshold = greedy_improvement_threshold
        self.bag_predicates = bag_predicates
        self.bag_features = bag_features

        if not self.randomize_feature_order:
            self.tiebreak_counter = itertools.count()

    def __str__(self):
        if PRINT_MODE == 'tree':
            return str(self.root).rstrip()
        return self.print_conditionals(self.root)

    @property
    def learning_time(self):
        return time.time() - self.start_time

    def init(self, X, Y):
        """
        Helper method for self.fit.
        See self.fit for parameters.
        """
        assert not self.is_fit

        # Internally we store a map from sample ID to input/output data
        X = dict(enumerate(X))
        Y = dict(enumerate(Y))

        # Parse all of the features that appear in the input
        self.features = set()
        for x in X.values():
            self.features.update(x.keys())
        self.features = sorted(list(self.features))

        # Parse all of the classes that appear in the outputs
        self.classes = set()
        for y in Y.values():
            if isinstance(y, list):
                self.classes.update(y)
            else:
                self.classes.add(y)
        self.classes = sorted(list(self.classes))

        return X, Y

    def fit(self, X, Y):
        """
        Parameters
        ----------
        X : [{ str : bool }]
            Input data. Each datum is a map from feature names to values.
        Y : [ int ]
            Categorical target classes.
        """
        self.start_time = time.time()
        self.orig_X, self.orig_Y = X, Y

        # Internally we store a map from sample ID to input/output data
        self.X, self.Y = self.init(X, Y)

        # Initialize the tree to be an empty root
        root_idxs = sorted(list(self.X))
        root = self.LeafNode(None, None, root_idxs, None, False, False)
        root.score = self.get_score(root_idxs, root)

        # Edge case: trivial tree
        if root.score == 0.:
            #root.leaf_class = self.find_omnipresent_class(root_idxs, root)
            root.leaf_class = self.find_best_class(root_idxs, root)
            self.root = root
            self.is_fit = True
            return

        # Find a subtree that improves on the initial tree
        if self.root_feature is None:
            self.root = self.search_for_subtree(root)
            # import ipdb; ipdb.set_trace()
        elif isinstance(self.root_feature, list):
            node = root
            for feature in self.root_feature:
                new_node = self.split_node(node, feature)
                self.commit_node(new_node)
                if new_node.parent:
                    new_node.parent.right = new_node
                node = new_node.right
            self.root = node.root
        else:
            self.root = self.split_node(root, self.root_feature)

        # Recursively split from the root
        self.split(self.root, 1)

        self.is_fit = True

    def predict(self, x, ret_node=False):
        """
        Predict the class of a single input.
        Parameters
        ----------
        x : { str : bool }
            A map from feature names to values.
        Returns
        -------
        prediction : int
            The predicted class.
        """
        assert self.is_fit, "Must call is_fit before predict."

        # Create a temporary sample ID for the given input
        sample_id = 1 if len(self.X) == 0 else max(self.X.keys()) + 1
        self.X[sample_id] = x

        # Make prediction
        predicted_node = self.predict_from_node(self.root, sample_id)

        # Delete temporary sample ID
        del self.X[sample_id]

        if ret_node:
            return predicted_node

        return predicted_node.leaf_class

    def predict_from_node(self, node, idx):
        """
        Predict the class of an input starting at the given node.
        Helper function for self.predict.
        Parameters
        ----------
        node : {str : Any} or int
            See class docstring.
        idx : int
            Sample ID for an input.
        Returns
        -------
        prediction : Node
            The predicted node.
        """
        if node.is_leaf:
            return node

        if self.feature_holds(node.feature, idx, node):
            return self.predict_from_node(node.right, idx)

        return self.predict_from_node(node.left, idx)

    def search_for_subtree(self, node):
        """
        Find a tree that splits the data and results in a score improvement.
        Parameters
        ----------
        node : Node
        Returns
        -------
        subtree : Node
            An expanded version of the input node.
        """
        if DEBUG:
            print("Searching for subtree from")
            print(node)

        # Check whether we already have a perfect leaf
        if node.score == 0.:
            return node

        # Goal is a tree that improves on the score
        root_score = node.score
        if DEBUG:
            print("** ROOT SCORE **", root_score)

        # Queue of subtrees
        queue = [(0, root_score, 0, node)]

        # Track best subtree found if we need to give up
        best_subtree_found = None
        best_subtree_score = np.inf

        # Perform BFS
        while len(queue) > 0:
            priority, score, tiebreak, subtree = hq.heappop(queue)

            if score < best_subtree_score:
                best_subtree_found = subtree
                best_subtree_score = score

            # Give up before perfect split
            if priority > self.max_feature_length or self.learning_time > self.max_learning_time:
                if self.max_exceeded_strategy == 'early_stopping':
                    return best_subtree_found
                elif self.max_exceeded_strategy == 'fail':
                    raise LearningFailure()
                elif self.max_exceeded_strategy == 'pdb':
                    import ipdb; ipdb.set_trace()
                raise Exception("Unknown max exceeded strategy '{}'.".format(self.max_exceeded_strategy))

            if DEBUG:
                print("\tPopped subtree (id={}) ".format(subtree._hash) + 
                      "with feature priority {} and score {}".format(priority, score))
                print(subtree)

            # Subtree found; we're done
            if root_score - score > self.greedy_improvement_threshold:
                # TEMPORARY HACK?
                if ((priority > 1 or score == 0.) and
                   # ANOTHER HACK: verify that this subtree didn't add an action predicate
                    all((desc.full_feature_path[0].predicate !=
                         desc.full_feature_path[-1].predicate)
                        for desc in subtree.descendents
                        if isinstance(desc.full_feature_path[0], Literal))):
                    if DEBUG:
                        print("Found node with score {} (vs {}):".format(score, root_score))
                        print(subtree)
                    return subtree

            # Find next subtrees
            for child in self.get_subtree_expansions(subtree):

                # if DEBUG:
                #     print("\t\tchild subtree (id={}):".format(child._hash))
                #     print(child)
                #     print("\t\twith score")
                #     print("\t\t", child.score)
                #     print("\t\tand priority")
                #     print("\t\t", priority)

                if self.randomize_feature_order:
                    tiebreak = self.random.uniform()
                else:
                    tiebreak = next(self.tiebreak_counter)

                # Finish early if there's a perfect split, because no way we'll do
                # better
                if child.score == 0.:
                    hq.heappush(queue, (0, child.score, tiebreak, child))
                    break

                hq.heappush(queue, (priority + 1, child.score, tiebreak, child))

        print("No split found.")

        # Add best leaf classes
        if node.is_leaf:
            node.leaf_class = self.find_best_class(node.idxs, node)
        else:
            for leaf in node.leaf_nodes:
                leaf.leaf_class = self.find_best_class(leaf.idxs, leaf)

        return node

    def get_subtree_expansions(self, subtree):
        """
        Given a subtree, find all possible single-step expansions
        of the tree.
        Parameters
        ----------
        subtree : Node
        Returns
        -------
        subsubtrees : [Node]
        """
        # If this node is a leaf and has perfect score, no more
        # subtrees below
        if subtree.is_leaf and subtree.score == 0.:
            return []

        # If this node is not a leaf, find the first leaf to expand
        if not subtree.is_leaf:
            new_subtrees = []

            for subsubtree in self.get_subtree_expansions(subtree.right):
                # TODO: avoid copy
                new_subtree = subtree.copy()
                new_subtree.right = subsubtree
                subsubtree.parent = new_subtree
                new_subtrees.append(new_subtree)

            # It doesn't seem like this makes any difference.
            # TODO: investigate whether splitting down the left as well
            # helps.
            # if len(new_subtrees) > 0:
            #     return new_subtrees

            # for subsubtree in self.get_subtree_expansions(subtree.left):
            #     # TODO: avoid copy
            #     new_subtree = subtree.copy()
            #     new_subtree.left = subsubtree
            #     subsubtree.parent = new_subtree
            #     new_subtrees.append(new_subtree)

            return new_subtrees

        # This is a leaf, let's expand it by adding features
        features = self.get_features(subtree)

        if DEBUG:
            print("\tConsidering {} possible features".format(len(features)))

        children = []

        for feat_i, feature in enumerate(features):
            # No feature repeats
            if feature in subtree.full_feature_path:
                continue
            new_subtree = self.split_node(subtree, feature)
            children.append(new_subtree)

        return children

    def split_node(self, node, feature):
        """
        Split the node given the feature.
        Parameters
        ----------
        node : Node
        feature : str
        """
        # Get groups
        left_idxs, right_idxs = self.get_groups(node, feature)

        # Split the node
        node = node.split(feature, None, None, 1.0, 1.0, left_idxs, right_idxs)

        # Get scores
        score_left = self.get_score(left_idxs, node.left)
        score_right = self.get_score(right_idxs, node.right)

        # If score is perfect, find best class
        if score_left == 0.:
            class_left = self.find_best_class(left_idxs, node.left)
        else:
            class_left = None
        if score_right == 0.:
            class_right = self.find_best_class(right_idxs, node.right)
        else:
            class_right = None

        node.left.score = score_left
        node.right.score = score_right
        node.left.leaf_class = class_left
        node.right.leaf_class = class_right

        return node

    def split(self, node, depth):
        """
        Given a new node with a feature and split data, create child nodes.
        Parameters
        ----------
        node : dict or int
            See class docstring.
        depth : int
            The current depth in the tree of the node.
        """
        # Check for a no split
        if node.is_leaf:
            return

        # Give up
        if depth >= self.max_depth:
            import ipdb; ipdb.set_trace()

        # Commit node
        self.commit_node(node)

        # Recurse on child leaves
        node.right = self.split_helper(node.right, depth)
        node.left = self.split_helper(node.left, depth)

    def split_helper(self, node, depth):
        if node.is_leaf:
            # Don't recurse on perfect leaves
            if node.score == 0.:
                return node

            # Recursively find subtrees for left and right
            new_node = self.search_for_subtree(node)
            self.split(new_node, depth+1)
            return new_node

        else:
            node.right = self.split_helper(node.right, depth)
            node.left = self.split_helper(node.left, depth)
            return node

    def commit_node(self, node):
        """
        Mark a node as committed to the tree.
        Parameters
        ----------
        node : Node
        """
        node.committed = True
        for descendent in node.descendents:
            self.commit_node(descendent)

    def find_best_class(self, idxs, node, ret_score=False):
        """
        Find the best class for the idxs.
        Parameters
        ----------
        idxs : [ int ]
        node : DTNode
        ret_score : bool
        Returns
        -------
        class : [ Any ]
        likelihood : float
            If ret_score is True.
        """
        try:
            # Multi-output
            target_max_likelihoods = { idx : {uc : -10e8 for uc in self.Y[idx]} for idx in idxs }
            multioutput = True
        except TypeError:
            # Single output
            target_max_likelihoods = { idx : {self.Y[idx] : -10e8} for idx in idxs }
            multioutput = False

        num_targets = None
        for idx, targets in target_max_likelihoods.items():
            if num_targets is None:
                num_targets = len(targets)
            # Quick impossibility check
            elif num_targets != len(targets):
                if ret_score:
                    return None, 10e8
                return None

        final_class = set()

        for class_val in self.get_classes(node):

            cls_ll = 0.
            explained_targets = defaultdict(set)

            for idx in idxs:
                # Number of bindings for which preconditions hold
                pst = self.get_pst(node, idx)
                pre_bindings = pst.prove([],
                    max_assignment_count=MAX_ASSIGNMENT_COUNT)
                num_pre_bindings = len(pre_bindings)
                if num_pre_bindings == MAX_ASSIGNMENT_COUNT:
                    num_pre_bindings = 100000

                if DEBUG:
                    if num_pre_bindings == 0: import ipdb; ipdb.set_trace()

                # Number of bindings for which each class holds (typically 0 or 1)
                assignments = pst.prove(class_val, 
                    max_assignment_count=MAX_ASSIGNMENT_COUNT)
                num_cls_bindings = len(assignments)
                # if num_cls_bindings == MAX_ASSIGNMENT_COUNT:
                #     num_cls_bindings = 100000

                # Likelihood of the observed y given x
                if num_cls_bindings == 0:
                    idx_ll = -10e8
                else:
                    idx_ll = np.log(num_cls_bindings) - np.log(num_pre_bindings)
    
                cls_ll += idx_ll

                if num_cls_bindings > 0:
                    for assignment in assignments:
                        which_class_holds = self.ground_literal_from_assignments(
                            class_val, assignment)
                        explained_targets[idx].add(which_class_holds)

            for idx, targets in explained_targets.items():
                for target in targets:
                    if target not in target_max_likelihoods[idx]:
                        continue
                    current_ml = target_max_likelihoods[idx][target]
                    if cls_ll > current_ml:
                        target_max_likelihoods[idx][target] = cls_ll
                        final_class.add(class_val)

            if len(final_class) == num_targets:
                final_ll = 0.
                for idx in idxs:
                    for target in target_max_likelihoods[idx]:
                        ll = target_max_likelihoods[idx][target]
                        final_ll += ll
                # final_likelihood = np.exp(final_ll)
                # score = 1. - final_likelihood
                if np.isinf(final_ll):
                    final_ll = -10e8
                score = -final_ll
                if multioutput:
                    if ret_score:
                        return sorted(list(final_class)), score
                    else:
                        return sorted(list(final_class))
                if ret_score:
                    return list(final_class)[0], score
                else:
                    return list(final_class)[0]

        if ret_score:
            return None, 10e8
        return None

    def graph_tree(self, fname):
        """Graph tree using graphviz.
        """
        assert fname.endswith(".png")
        dot = Digraph(format="png")
        self._graph_tree_helper(dot, self.root, parent=None)
        dot.render(fname.replace(".png", ""))
        os.remove(fname.replace(".png", ""))
        print("Wrote out learned tree to {}".format(fname))

    def _graph_tree_helper(self, dot, node, parent):
        fn = lambda s : s.orig_str() if hasattr(s, 'orig_str') else str(s)
        if isinstance(node, DTNode):
            if isinstance(node.label, list): # leaf list
                name = "\n".join(map(fn, node.label))
            else:
                name = fn(node.label)
        elif node is None:
            name = "None" # failure
        else:
            name = fn(node)
        dot.node(name, name)
        if parent is not None:
            dot.edge(parent[0], name, label=parent[1])
        if not node.is_leaf:
            self._graph_tree_helper(dot, node.left, parent=(name, "False"))
            self._graph_tree_helper(dot, node.right, parent=(name, "True"))

    def get_conditional_literals(self, node):
        if node.is_leaf:
            return [([], node.label)]

        paths = []
        for (path, leaf) in self.get_conditional_literals(node.left):
            paths.append(([node.get_naf_label()] + path, leaf))

        for (path, leaf) in self.get_conditional_literals(node.right):
            paths.append(([node.label] + path, leaf))

        return paths

    def get_all_paths_to_leaves(self, node, colored):
        if node.is_leaf:
            return [([], node.label)]

        paths = []
        for (path, leaf) in self.get_all_paths_to_leaves(node.left, colored):
            feat_str = str(node.label) if colored else node.label.orig_str()
            paths.append((['~' + feat_str] + path, leaf))

        for (path, leaf) in self.get_all_paths_to_leaves(node.right, colored):
            feat_str = str(node.label) if colored else node.label.orig_str()
            paths.append(([feat_str] + path, leaf))

        return paths

    def print_conditionals(self, colored=True):
        s = ''
        paths_to_leaves = self.get_all_paths_to_leaves(self.root, colored=colored)
        for path, leaf in paths_to_leaves:
            if leaf:
                if colored:
                    leaf_str = str(leaf)
                elif isinstance(leaf, list):
                    leaf_str = str([s.orig_str() for s in leaf])
                else:
                    leaf_str = leaf.orig_str()
                s += ' & '.join(path) + ' => ' + leaf_str + '\n'
        return s

    def get_features(self, node):
        """
        Get all features associated with the given sample IDs.
        Also shuffle the order in which they are returned;
        this is important for randomization.
        Parameters
        ----------
        node : None
            Used by subclass.
        Returns
        -------
        features : [ str ]
            Features in random order.
        """
        if self.randomize_feature_order:
            self.random.shuffle(self.features)
        return self.features

    def get_classes(self, node):
        """
        Get all classes associated with the given sample IDs.
        Parameters
        ----------
        node : None
            Used by subclass.
        Returns
        -------
        classes : [ str ]
            Classes.
        """
        return self.classes

    def feature_holds(self, feature, idx, node):
        """
        Check whether a feature is True for the given sample ID.
        Parameters
        ----------
        node : Node
        idx : int
            The sample ID.
        Returns
        -------
        feature_holds : bool
        """
        return self.X[idx][feature]

    # def find_class_that_holds(self, class_val, idx, node):
    #     """
    #     Check whether a class has a value the given sample ID.

    #     Parameters
    #     ----------
    #     class_val : int
    #         The class value.
    #     idx : int
    #         The sample ID.
    #     node : Node

    #     Returns
    #     -------
    #     which_class_holds : int
    #     """
    #     if isinstance(self.Y[idx], list) and class_val in self.Y[idx]:
    #         return class_val
    #     elif self.Y[idx] == class_val:
    #         return class_val
    #     return None

    def get_groups(self, node, feature):
        """
        Split data based on whether a feature holds.
        Parameters
        ----------
        node : Node
        feature : str
        Returns
        -------
        groups : ([ int ], [ int ])
            Negative sample IDs, positive sample IDs.
        """
        left_group, right_group = [], []

        idxs = node.idxs

        for idx in idxs:
            if self.feature_holds(feature, idx, node):
                right_group.append(idx)
            else:
                left_group.append(idx)

        return (left_group, right_group)

    def get_score(self, idxs, node):
        if self.criterion.lower() == 'classification':
            return self.classification_criterion(idxs, node)
        raise NotImplementedError()

    def classification_criterion(self, group, node):
        """
        Multiclass, multi-output classification criterion.
        Parameters
        ----------
        group : [ int ]
        node : Node
        Returns
        -------
        score : float
            Lower is better. Between 0 and 1.
        """
        if len(group) == 0:
            return 0.

        _, score = self.find_best_class(group, node, ret_score=True)
        return score



class FOLDTNode(DTNode):
    """
    A node in a first-order logic decision tree.
    """
    @property
    def available_variables(self):
        available_variables = defaultdict(list)
        for literals in self.positive_feature_path:
            if not isinstance(literals, list):
                literals = [literals]
            for literal in literals:
                for var in literal.variables:
                    var_type = var.var_type
                    if var not in available_variables[var_type]:
                        available_variables[var_type].append(var)
        return available_variables


class FOLDTFeatureNode(DTFeatureNode, FOLDTNode):
    pass


class FOLDTLeafNode(DTLeafNode, FOLDTNode):
    FeatureNode = FOLDTFeatureNode


class FOLDTClassifier(DecisionTreeClassifier):
    """
    First Order Logic Decision Tree Classifier.
    A decision tree where the internal nodes and leaf nodes
    are ungrounded literals, e.g., At(Variable1, Variable2).
    An internal node (e.g. the root node) holds for an input
    if there is some binding of variables that satisfies the
    literals along the path from the root to the node.
    Input to the decision tree is a set of grounded literals.
    The output is one or more grounded literals (classes).
    """
    LeafNode = FOLDTLeafNode

    def init(self, X, Y):
        """
        Process the input data. Parse the predicates. Set up the provers.
        Parameters
        ----------
        X : [{ Literal }]
            Each input is a set of ground literals.
        Y : [ Literal ]
            Outputs are single ground literal classes.
        """
        X = [self.preprocess_literal_set(x) for x in X]
        X, Y = DecisionTreeClassifier.init(self, X, Y)

        # For clarity, we will not use 'features' here, but predicates
        self.predicates = self.features
        delattr(self, 'features')
        if self.bag_predicates:
            # Predicate bagging: if we have K predicates, then bag by
            # sampling K predicates with replacement, and only considering those
            # for the entire tree-learning process.
            inds = set(self.random.choice(len(self.predicates), size=len(self.predicates)))
            self.predicates = [self.predicates[ind] for ind in inds]

        self.next_variable_id = itertools.count()

        self.X, self.Y = X, Y

        return X, Y

    def preprocess_literal_set(self, s):
        """
        Convert a set of ground literals to a dictionary
        mapping predicate name to literal lists.
        Parameters
        ----------
        s : { Literal }
        Returns
        -------
        d : { str : [ Literal ] }
        """
        d = {}

        for lit in s:
            if lit.predicate not in d:
                d[lit.predicate] = []
            d[lit.predicate].append(lit)

        return d

    def get_features(self, node):
        """
        Get all features associated with the given sample IDs.
        Note that features are ground literals for FOLDT!
        The available features for a set of indices will be
        all predicates in the samples with all possible
        variable orderings. Placeholder variables are introduced
        and later substituted if the feature is selected.
        Also shuffle the order in which they are returned;
        this is important for randomization.
        Parameters
        ----------
        node : Node
        Returns
        -------
        features : [ str ]
            Features in random order.
        """
        # All the names variables established so far for these samples
        available_variables = node.available_variables

        out = set()

        for predicate in self.predicates:
            for variable_settings in self.iter_variable_settings(available_variables, predicate):
                out.add(predicate(*variable_settings))
        out = sorted(list(out))

        if self.bag_features:
            # Feature bagging: if we have K features, then bag by
            # sampling K features with replacement, and only considering those
            # for this split only.
            inds = set(self.random.choice(len(out), size=len(out)))
            out = [out[ind] for ind in inds]
        if self.randomize_feature_order:
            self.random.shuffle(out)
        return out

    def iter_variable_settings(self, available_variables, predicate, placeholders_allowed=True):
        """
        Helper method for get_features and get_classes.
        Parameters
        ----------
        available_variables : { None or Type : [ str ] }
        predicate : Predicate
        placeholders_allowed : bool
        Yields
        ------
        variable_settings : tuple(str)
        """
        if predicate.var_types is None:
            var_types = [NULLTYPE for _ in range(predicate.arity)]
        else:
            var_types = predicate.var_types

        if placeholders_allowed:
            entrywise_variable_possibilities = [ available_variables[vt] + ["Placeholder"] \
                for vt in var_types]
        else:
            entrywise_variable_possibilities = [ available_variables[vt] \
                for vt in var_types]

        # Get the max_placeholder_id in available_variables
        max_placeholder_id = 0
        for vs in available_variables.values():
            for v in vs:
                if 'Placeholder' in v:
                    placeholder_id = int(v.split('Placeholder')[1].split(':')[0])
                    max_placeholder_id = max(max_placeholder_id, placeholder_id)

        for variables in itertools.product(*entrywise_variable_possibilities):
            # Handle placeholders
            new_variables = []
            placeholder_count = max_placeholder_id + 1
            for i, v in enumerate(variables):
                if 'Placeholder' == v:
                    v = v + str(placeholder_count)
                    if var_types[i] is not None:
                        v = var_types[i](v)
                    else:
                        v = NULLTYPE(v)
                    placeholder_count += 1
                new_variables.append(v)
            variables = new_variables

            # No repeats allowed
            if len(variables) != len(set(variables)):
                continue

            yield variables

    def commit_node(self, node):
        """
        Mark a node as committed to the tree.
        Parameters
        ----------
        node : Node
        """
        # Replace placeholders
        self.replace_placeholders(node)

        # Mark as committed
        super().commit_node(node)

    def replace_placeholders(self, node, newly_created_variables=None):
        """
        Replace all Placeholders with new variables in a node and its
        descendents.
        Parameters
        ----------
        node : Node
        newly_created_variables : { str : str }
            Placeholder : new variable.
        """
        if node.is_leaf:
            literals = node.leaf_class
        else:
            literals = node.feature

        if literals is None:
            return

        if not isinstance(literals, list):
            literals = [literals]

        if newly_created_variables is None:
            newly_created_variables = {}

        # Replace placeholders
        for lit_idx, literal in enumerate(literals):
            new_literal_variables = []
            for variable in literal.variables:
                if "Placeholder" in variable:
                    if variable in newly_created_variables and \
                        newly_created_variables[variable] in \
                            node.available_variables[variable.var_type]:
                        new_variable = newly_created_variables[variable]
                    else:
                        var_type = variable.var_type
                        new_variable = var_type("?Var{}".format(
                            next(self.next_variable_id)))
                        newly_created_variables[variable] = new_variable
                else:
                    new_variable = variable
                new_literal_variables.append(new_variable)

            if node.is_leaf:
                if isinstance(node.leaf_class, list):
                    node.leaf_class[lit_idx].set_variables(new_literal_variables)
                else:
                    assert lit_idx == 0
                    node.leaf_class.set_variables(new_literal_variables)
            else:
                if isinstance(node.feature, list):
                    node.feature[lit_idx].set_variables(new_literal_variables)
                else:
                    assert lit_idx == 0
                    node.feature.set_variables(new_literal_variables)

        for descendent in node.descendents:
            self.replace_placeholders(descendent, 
                newly_created_variables=newly_created_variables)

    def get_classes(self, node):
        """
        Get all classes associated with the given sample IDs.
        The available classes for a set of indices will be
        all predicates in the samples with all possible
        variable orderings. No placeholders allowed!
        Parameters
        ----------
        node : Node
        Returns
        -------
        classes : [ Literal ]
        """
        available_variables = node.available_variables

        out = set()

        for literal in self.classes:
            predicate = literal.predicate
            variable_settings = self.iter_variable_settings(available_variables, predicate, 
                placeholders_allowed=False)
            for variables in variable_settings:
                out.add(predicate(*variables))
        out = sorted(list(out))

        return out

    def feature_holds(self, feature, idx, node, ret_assignments=False):
        """
        Check whether a feature holds for the given sample ID.
        Parameters
        ----------
        feature : str
            The feature name.
        idx : int
            The sample ID.
        node : Node
        ret_assignments : bool
        Returns
        -------
        feature_holds : bool
        """
        try:
            pst = self.get_pst(node, idx)
            assignments = pst.prove(feature)
            feature_holds = (len(assignments) > 0)
        except CommitGoalError:
            feature_holds = False
            assignments = []
        if ret_assignments:
            return feature_holds, assignments
        return feature_holds

    # def find_class_that_holds(self, class_val, idx, node):
    #     """
    #     Check whether a class holds with the given sample ID.

    #     Parameters
    #     ----------
    #     class_val : int
    #         The class value.
    #     idx : int
    #         The sample ID.
    #     node : Node

    #     Returns
    #     -------
    #     which_class_holds : Literal or None
    #     """
    #     pst = self.get_pst(node, idx)
    #     import ipdb; ipdb.set_trace()
    #     if len(pst.prove(class_val.negate_as_failure())) > 0:
    #         return None
    #     # Get a positive binding
    #     assignments = pst.prove(class_val)
    #     if len(assignments) == 0:
    #         return None
    #     return self.ground_literal_from_assignments(class_val, assignments[0])

    def predict_from_node(self, node, idx, assignments=None):
        """
        Predict the class of an input starting at the given node.
        Helper function for self.predict.
        Parameters
        ----------
        node : {str : Any} or int
            See class docstring.
        idx : int
            Sample ID for an input.
        Returns
        -------
        prediction : Node
            The predicted node.
        assignments : None or dict
            Last variable assignments.
        """
        if node.is_leaf:
            return node, assignments

        holds, new_assignments = self.feature_holds(node.feature, 
            idx, node, ret_assignments=True)

        if holds:
            return self.predict_from_node(node.right, idx,
                assignments=new_assignments)

        return self.predict_from_node(node.left, idx,
            assignments=assignments)

    def predict(self, x, ret_node=False):
        """
        Predict the class of a single input.
        Parameters
        ----------
        x : { Literals }
            A set of ground literals
        Returns
        -------
        prediction : Literal
            The predicted class.
        """
        x = self.preprocess_literal_set(x)

        # Temporarily assign a sample ID to the input
        idx = 1 if len(self.X) == 0 else max(self.X.keys()) + 1
        self.X[idx] = x

        # Predict the (ungrounded) class
        predicted_node, assignments = self.predict_from_node(self.root, idx)

        # Delete temporary idx
        del self.X[idx]

        if ret_node:
            return predicted_node

        # If None, give up
        if predicted_node.leaf_class is None:
            return None

        if isinstance(predicted_node.leaf_class, list):
            if assignments is None:
                assignments = [{}]
            prediction = [self.ground_literal_from_assignments(l, assignments[0]) \
                for l in predicted_node.leaf_class]
        else:
            if assignments is None:
                assignments = [{}]
            prediction = self.ground_literal_from_assignments(predicted_node.leaf_class, 
                assignments[0])

        return prediction

    def get_pst(self, node, idx):
        """
        Parameters
        ----------
        node : Node
        idx : int
        """
        x = self.X[idx]

        # The knowledge base of the prover contains the ground literals 
        # from the sample
        kb = [literal for pred in sorted(x.values()) for literal in pred]

        if idx in self.Y:
            y = self.Y[idx]
            if isinstance(y, list):
                kb.extend(y)
            else:
                kb.append(y)

        pst = ProofSearchTree(kb,
            allow_redundant_variables=False,
            allow_commit_exception=self.allow_commit_exception)

        # The goal_literals include all of the positive literals in the
        # path to the root
        for lit in node.positive_feature_path:
            pst.commit_goal(lit)

        return pst

    def ground_literal_from_assignments(self, literal, assignments):
        grounding = []
        for v in literal.variables:
            if v not in assignments:
                import ipdb; ipdb.set_trace()
                raise Exception("Predicted a literal class without a binding.")

            grounding.append(assignments[v])

        return Literal(literal.predicate, grounding)
