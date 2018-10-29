from collections import defaultdict
from time import time
import vrgs.full_info as full_info
import vrgs.part_info as part_info
import vrgs.no_info as no_info

class VRG:
    """
    Class for Vertex Replacement Grammars
    """
    def __init__(self, mode, k, selection):
        self.mode = mode  # type of VRG - full, part, or no
        self.k = k
        self.selection = selection  # selection strategy - random, mdl, level, or mdl_levels

        self.rule_list = []   # list of rule objects
        self.rule_dict = defaultdict(list)  # dictionary of rules, keyed in by their LHS
        self.mdl = 0  # the MDL of the rules

    def __len__(self):
        return len(self.rule_list)

    def __contains__(self, item):
        return item in self.rule_dict[item.lhs]

    def __str__(self):
        if self.mdl == 0:
            self.calculate_cost()
        return '{} {} {} {} rules, {} bits'.format(self.k, self.mode, self.selection, len(self.rule_list), self.mdl)

    def add_rule(self, rule):
        # adds to the grammar iff it's a new rule
        isomorphic = False
        for old_rule in self.rule_dict[rule.lhs]:
            if rule == old_rule:  # check for isomorphism
                old_rule.frequency += 1
                isomorphic = True
                break

        if not isomorphic:
            self.rule_list.append(rule)  # add the rule to the list of rules
            self.rule_dict[rule.lhs].append(rule)  # add the rule to the rule dictionary

    def calculate_cost(self):
        for rule in self.rule_list:
            rule.calculate_cost()
            self.mdl += rule.cost

    def get_cost(self):
        # if self.mdl != 0:  # the cost has been computed before
        #     return self.mdl
        # else:
        self.mdl = 0
        self.calculate_cost()
        return self.mdl

    def generate_graphs(self, count):
        """
        generate count many graphs from the grammar
        :param count: number of graphs to be generated
        :return:
        """
        if self.mode == 'full':
            generate_graph = full_info.generate_graph
        elif self.mode == 'part':
            generate_graph = part_info.generate_graph
        else:
            generate_graph = no_info.generate_graph

        graphs = []

        for _ in range(count):
            start_time = time()
            h = generate_graph(self.rule_dict)
            t = time() - start_time
            graphs.append(h)
            print('n = {} m = {} ({} secs)'.format(h.order(), h.size(), round(t, 3)))

        return graphs