from collections import defaultdict
from time import time
import os

import src.full_info as full_info
import src.part_info as part_info
import src.no_info as no_info

class VRG:
    """
    Class for Vertex Replacement Grammars
    """
    def __init__(self, mode, selection, clustering, name, lamb=None):
        self.name = name  # name of the graph
        self.mode = mode  # type of VRG - full, part, or no
        self.lamb = lamb
        self.selection = selection  # selection strategy - random, mdl, level, or mdl_levels
        self.clustering = clustering  # clustering strategy

        self.rule_list = []   # list of Rule objects
        self.rule_dict = defaultdict(list)  # dictionary of rules, keyed in by their LHS
        self.mdl = 0  # the MDL of the rules

    def __len__(self):
        return len(self.rule_list)

    def __contains__(self, item):
        return item in self.rule_dict[item.lhs]

    def __str__(self):
        if self.mdl == 0:
            self.calculate_cost()
        return 'mode: {} clustering: {} selection: {} lambda: {} rules: {} mdl: {} bits'.format(self.mode, self.clustering, self.selection,
                                                      self.lamb, len(self.rule_list), round(self.mdl, 3))

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

        if not os.path.exists(f'./output/rule_orders/{self.name}'):
            os.makedirs(f'./output/rule_orders/{self.name}')

        for i in range(count):
            # start_time = time()
            h, rule_ordering = generate_graph(self.rule_dict, self.rule_list)

            with open(f'./output/stats/{self.name}_{self.clustering}_{self.selection}_{self.lamb}_sizes.txt', 'a') as f:
                f.write(f'{h.order()}, {h.size()}\n')
                # f.write(','.join(map(str, rule_ordering)) + '\n')

            # t = time() - start_time
            graphs.append(h)
            # print('n = {} m = {} ({} secs)'.format(h.order(), h.size(), round(t, 3)))

        return graphs