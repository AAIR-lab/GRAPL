import logging
import time

from .command import count_file_lines, remove_duplicate_lines, read_file


class CNFWriter:
    def __init__(self, filename):
        self.filename = filename
        self.variables = dict()
        self.clauses = set()
        self.clause_batch = []
        self.num_clauses = 0
        self.accumulated_weight = 0
        self.mapping = dict()
        self.closed = False
        self.variable_index = None

        self.buffer_filename = self.filename + ".tmp"
        self.buffer = open(self.buffer_filename, "w")

    def variable(self, name):
        assert not self.closed
        return self.variables.setdefault(name, Variable(name))

    def literal(self, variable, polarity):
        assert self.closed
        i = self.variable_index[variable]
        return i if polarity else -1 * i

    def clause(self, literals, weight=None):
        assert self.closed
        assert weight is None or weight > 0
        # self.clauses.add(Clause(literals=literals, weight=weight)) # Keeping the set in memory is expensive!
        self.num_clauses += 1
        self.accumulated_weight += weight if weight is not None else 0

        self.clause_batch.append((literals, weight))
        if len(self.clause_batch) == 1000:
            self.flush_clauses()

    def flush_clauses(self):
        clauses_str = '\n'.join(print_clause(literals, weight) for literals, weight in self.clause_batch)
        print(clauses_str, file=self.buffer)
        del self.clause_batch
        self.clause_batch = []

    def save(self):
        assert self.closed
        numvars = len(self.variables)
        numclauses = self.num_clauses

        self.flush_clauses()
        self.buffer.close()
        self.buffer = None
        self._save(self.filename, numvars, numclauses)

        # debug = True
        # debug = False
        # if debug:
        #     dfilename = "{}.txt".format(self.filename)
        #     debug_clause_printer = lambda c: str(c)
        #     self._save(dfilename, numvars, numclauses, top, debug_clause_printer)

    def _save(self, filename, numvars, numclauses):
        num_written_clauses = count_file_lines(self.buffer_filename)
        assert numclauses == num_written_clauses
        # remove_duplicate_lines(self.buffer_filename)
        num_unique_clauses = count_file_lines(self.buffer_filename)
        # num_unique_clauses_in_mem = len(self.clauses)
        # assert num_unique_clauses == num_unique_clauses_in_mem  # Keeping the set in memory is expensive!
        top = str(self.accumulated_weight + 1)

        logging.info("Writing max-sat encoding to file \"{}\"".format(self.filename))
        logging.info("Max-sat problem: {} vars and {} unique clauses (with repetitions: {}). Top weight: {}".format(
            numvars, num_unique_clauses, numclauses, top))

        with open(filename, "w") as output:
            print("c WCNF model generated on {}".format(time.strftime("%Y%m%d %H:%M:%S", time.localtime())),
                  file=output)
            # p wcnf nbvar nbclauses top
            print("p wcnf {} {} {}".format(numvars, num_unique_clauses, top), file=output)
            for line in read_file(self.buffer_filename):
                print(line.replace("TOP", top), file=output)
                # for clause in self.clauses:
                #     print(clause_printer(clause), file=file)

    def close(self):  # Once closed, the writer won't admit more variable declarations
        assert not self.closed
        self.closed = True
        self.variable_index = {var: i for i, var in enumerate(self.variables.values(), start=1)}
        # Save the variable mapping to parse the solution later
        self.mapping = {i: name for name, i in self.variable_index.items()}

    def print_variables(self, filename):
        assert self.closed
        variables = sorted(self.variable_index.items(), key=lambda x: x[1])  # Sort by var index
        with open(filename, 'w') as f:
            print("\n".join("{}: {}".format(i, v) for v, i in variables), file=f)


class Variable:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    __repr__ = __str__


class Literal:
    def __init__(self, variable, polarity=True):
        assert isinstance(variable, Variable)
        self.variable = variable
        self.polarity = polarity

    def __neg__(self):
        return Literal(self.variable, not self.polarity)

    def __str__(self):
        return "{}{}".format(("" if self.polarity else "-"), self.variable)

    def to_cnf(self, variable_index):
        return "{}{}".format(("" if self.polarity else "-"), variable_index[self.variable])

    def __eq__(self, other):
        return self.variable == other.variable and self.polarity == other.polarity

    def __hash__(self):
        return hash((self.variable, self.polarity))

    __repr__ = __str__


class Clause:
    def __init__(self, literals, weight=None):
        # assert all(isinstance(l, Literal) for l in literals)
        # self.literals = tuple(literals)
        self.literals = literals
        # assert len(set(literals)) == len(self.literals)  # Make sure all literals are unique
        self.weight = weight

    def __str__(self):
        return "{{{}}} [{}]".format(','.join(str(l) for l in self.literals), self.weight)

    __repr__ = __str__

    # def cnf_line(self, top, variable_index):
    #     # w <literals> 0
    #     w = top if self.weight is math.inf else self.weight
    #     literals = " ".join(l.to_cnf(variable_index) for l in self.literals)
    #     return "{} {} 0".format(w, literals)

    # def cnf_line(self, top):
    #     # w <literals> 0
    #     w = top if self.weight is math.inf else self.weight
    #     literals = " ".join(str(l) for l in self.literals)
    #     return "{} {} 0".format(w, literals)

    # def __eq__(self, other):
    #     return self.literals == other.literals
    #
    # def __hash__(self):
    #     return hash(self.literals)


def print_clause(literals, weight):
    # w <literals> 0
    w = "TOP" if weight is None else weight
    literals = " ".join(str(l) for l in literals)
    return "{} {} 0".format(w, literals)
