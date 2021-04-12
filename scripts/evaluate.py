import sound_law.rl.rule as rule
from pathlib import Path
from dev_misc import add_argument, g

if __name__ == "__main__":
    add_argument("calc_metric", dtype=bool, default=False, msg="Whether to calculate the metrics.")
    add_argument("out_path", dtype=str, msg="Path to the output file.")

    manager, gold, states, refs = rule.simulate()
    initial_state = states[0]
    if g.in_path:
        assert len(gold) == len(states) - 1
        if g.out_path:
            with Path(g.out_path).open('w', encoding='utf8') as fout:
                for state in states:
                    fout.write(f'{state.dist}\n')

    if g.calc_metric:
        # compute the similarity between the candidate ruleset and the gold standard ruleset
        candidate: List[SoundChangeAction] = None  # let this be the model's ruleset, which we are comparing to gold
        # first, what % of the gold ruleset is present in candidate?
        n_shared_actions = 0
        n_similar_actions = 0  # similar actions get half credit. We count separately so these are stored as int
        # TODO(djwyen) weight "partial credit" based on how similar the effects of the rules are, which can be calculated off distance
        for action in gold:
            similar_actions = manager.action_space.get_similar_actions(action)
            for candidate_act in candidate:
                if candidate_act == action:
                    n_shared_actions += 1
                if candidate_act in similar_actions:
                    n_similar_actions += 1
        ruleset_containment = (n_shared_actions + (.5 * n_similar_actions)) / len(gold)
        print('candidate ruleset contains ' + str(ruleset_containment) + '\% of the gold rules')
        # is there a way to combine this metric with the one below? i.e., to say that a given rule is only 'partially contained' within candidate if it's present, but in the wrong order relative to other dependent actions [actions it could feed or bleed]?

        # assume that candidate is a ruleset that contains the same rules as the gold ruleset but has them in a different order.
        # first, identify which pairs of actions are in the wrong order in candidate
        act_to_index = {act: i for i, act in enumerate(gold)}
        swaps = 0  # number of pairs that are out of order in an impactful way
        current_state = initial_state
        # assuming actions are applied in the order 0 to end
        for i, act1 in enumerate(candidate):
            for act2 in candidate[i + 1:]:
                if act_to_index[act1] > act_to_index[act2]:
                    # we do the checks in this order because the below is more computationally intensive than the above
                    if rule.order_matters(act1, act2, current_state):
                        swaps += 1
            current_state.apply_action(gold[i])  # update current state using gold action
        print(str(swaps) + ' pairs of rules in wrong order in candidate')
        # TODO two improvements for this metric:
        # 1. we currently use initial_state to test ordering, but should the state that we test the orderings on not evolve as we apply more rules? it may be that the context where the ordering matters only comes up after some rules have been applied; or that by the time the 2 rules in question are applied, any contexts such that the order matters are destroyed. So maybe each step we should change the state by the current action.
        # 2. there may be situations where A->B->C, that is that the ordering of A and B matter and the ordering of B and C matter, but the ordering of A and C do not matter. However, because of how the relationship is, we should care about the relative ordering of A and C; but this model won't detect that directly. One counterpoint: if A and C are in the wrong order, then at least one of them is in the wrong order relative to B, so it will be detected. But even still, perhaps the penalty should be higher than the penalty you normally accrue for swapping two rules, as the ordering of A and C are also wrong.
        # a complicated fix to the above would be to make a graph of actions and then use that to discover if a path A to C exists (and that therefore relative order matters) but it's unclear if this could be done for each of the n choose 2 ~ O(n^2) pairs of rules in a computationally efficient matter. One slightly good thing is that you can memoize to reduce the amount of searches to perhaps visiting each vertex exactly once.

        # greedily match up to 3 rules to each rule in gold
        threshold = 10  # amount a rule must decrease the distance by to be included in a poly-matching
        unmatched_rule_indices = set(range(len(candidate)))
        rule_pairings = {}  # maps the index of a rule in gold to a list of indices of its matched rules in candidate
        current_state = initial_state
        for i, act1 in enumerate(gold):
            next_state = current_state.apply_action(act1)
            next_segments = next_state.segments
            rule_pairings[i] = []

            # greedily identify up to 3 rules in candidate to be matched to rule i
            lowest_dist = None
            lowest_dist_ind = None
            last_rule_ind = -1  # rules are ordered, so we can only consider appending rules that come after the last picked rule
            cand_base_state = current_state  # the state resulting after all current matched rules are applied
            # distance resulting after all current matched rules are applied
            base_dist = current_state.dist_from(next_segments)
            for j in range(3):
                for ind in filter(lambda x: x > last_rule_ind, unmatched_rule_indices):
                    act2 = candidate[ind]
                    dist = cand_base_state.apply_action(act2).dist_from(next_segments)
                    if lowest_dist is None or (dist < lowest_dist and dist <= base_dist - threshold):
                        lowest_dist = dist
                        lowest_dist_ind = ind

                if lowest_dist_ind is None:  # no match found
                    break

                rule_pairings[i].append(lowest_dist_ind)
                unmatched_rule_indices.remove(lowest_dist_ind)
                last_rule_ind = lowest_dist_ind
                cand_base_state = cand_base_state.apply_action(candidate[lowest_dist_ind])
                base_dist = cand_base_state.dist_from(next_segments)

            current_state = next_state

        # evaluate how similar each rule in the pairing is by evaluating each of the paired rules and comparing how similar the results are
        # TODO
