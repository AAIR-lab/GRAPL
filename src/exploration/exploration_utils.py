import random

def get_action_lists(actions,
                     state, action_cache, model,
                     shuffle=True):

    learned_actions_to_try = []
    unlearned_actions_to_try = []
    for a in actions:

        action_name = a.predicate.name
        if action_name in action_cache:

            learned_actions_to_try.append(a)
        else:
            unlearned_actions_to_try.append(a)

    if shuffle:
        random.shuffle(learned_actions_to_try)
        random.shuffle(unlearned_actions_to_try)

    return learned_actions_to_try, \
        unlearned_actions_to_try
