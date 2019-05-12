'''
refactored VRG
'''

import os
from collections import defaultdict

import src.full_info as full_info
import src.no_info as no_info
import src.part_info as part_info
from typing import List, Dict
from src.Rule import PartRule

class VRG:
    """
    Class for Vertex Replacement Grammars
    """
    __slots__ = 'name', 'type', 'clustering', 'mu', 'rule_list', 'rule_dict', 'cost', 'num_rules'

    def __init__(self, type, clustering, name, mu):
        self.name: str = name  # name of the graph
        self.type: str = type  # type of grammar - lambda, local, global, selection strategy - random, dl, level, or dl_levels
        self.clustering: str = clustering  # clustering strategy
        self.mu = mu

        self.rule_list: List[PartRule] = []   # list of Rule objects
        self.rule_dict: Dict[int, List[PartRule]] = {}  # dictionary of rules, keyed in by their LHS
        self.cost:int = 0  # the MDL of the rules
        self.num_rules:int = 0  # number of active rules

    def copy(self):
        vrg_copy = VRG(type=self.type, clustering=self.clustering, name=self.name, mu=self.mu)
        vrg_copy.rule_list = self.rule_list[: ]
        vrg_copy.rule_dict = dict(self.rule_dict)
        vrg_copy.cost = self.cost
        vrg_copy.num_rules = self.num_rules
        return vrg_copy

    def __len__(self):
        return len(self.rule_list)

    def __contains__(self, rule: PartRule):
        return rule in self.rule_dict[rule.lhs]

    def __str__(self):
        if self.cost == 0:
            self.calculate_cost()
        st = f'graph: {self.name}, mu: {self.mu}, type: {self.type} clustering: {self.clustering} rules: {len(self.rule_list):_d}' \
            f'({self.num_rules:_d}) mdl: {round(self.cost, 3):_g} bits'
        return st
        # return f'{self.name}, mode: {self.mode} clustering: {self.clustering} selection: {} lambda: {} rules: {}({}) mdl: {} bits'.format(self.name, self.mode, self.clustering, self.selection,
        #                                                                                                 self.lamb, self.active_rules, len(self.rule_list), round(self.cost, 3))

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        return self.rule_list[item]

    def reset(self):
        # reset the grammar
        self.rule_list = []
        self.rule_dict = {}
        self.cost = 0
        self.num_rules = 0

    def add_rule(self, rule: PartRule) -> int:
        # adds to the grammar iff it's a new rule
        if rule.lhs not in self.rule_dict:
            self.rule_dict[rule.lhs] = []

        for old_rule in self.rule_dict[rule.lhs]:
            if rule == old_rule:  # check for isomorphism
                old_rule.frequency += 1
                rule.id = old_rule.id
                return old_rule.id


        # if I'm going to allow for deletions, there needs to be a better way to number things to prevent things from getting clobbered
        rule.id = self.num_rules
        # new rule
        self.num_rules += 1

        self.rule_list.append(rule)
        self.rule_dict[rule.lhs].append(rule)
        return rule.id

    # def deactivate_rule(self, rule_id):
    #     """
    #     deletes the rule with rule_id from the grammar
    #     :param rule_id:
    #     :return:
    #     """
    #     # do not decrease num_rules
    #     rule = self.rule_list[rule_id]
    #     rule.deactivate()
    #     # TODO check if rule deactivation propagates to the dictionary
    #     # self.rule_dict[rule.lhs]

    def calculate_cost(self):
        for rule in self.rule_list:
            rule.calculate_cost()
            self.cost += rule.cost
