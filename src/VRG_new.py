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
    __slots__ = 'name', 'type', 'clustering', 'rule_list', 'rule_dict', 'cost', 'num_active_rules'

    def __init__(self, type, clustering, name):
        self.name: str = name  # name of the graph
        self.type: str = type  # type of grammar - lambda, local, global, selection strategy - random, dl, level, or dl_levels
        self.clustering: str = clustering  # clustering strategy

        self.rule_list: List[PartRule] = []   # list of Rule objects
        self.rule_dict: Dict[int, List[PartRule]] = defaultdict(list)  # dictionary of rules, keyed in by their LHS
        self.cost = 0  # the MDL of the rules
        self.num_active_rules = 0  # number of active rules

    def __len__(self):
        return len(self.rule_list)

    def __contains__(self, rule):
        return rule in self.rule_dict[rule.lhs]

    def __str__(self):
        if self.cost == 0:
            self.calculate_cost()
        st = f'graph: {self.name}, type: {self.type} clustering: {self.clustering} rules: {len(self.rule_list)}({self.num_active_rules}) mdl: {round(self.cost, 3)} bits'
        return st
        # return f'{self.name}, mode: {self.mode} clustering: {self.clustering} selection: {} lambda: {} rules: {}({}) mdl: {} bits'.format(self.name, self.mode, self.clustering, self.selection,
        #                                                                                                 self.lamb, self.active_rules, len(self.rule_list), round(self.cost, 3))

    def add_rule(self, rule: PartRule) -> int:
        # adds to the grammar iff it's a new rule
        for old_rule in self.rule_dict[rule.lhs]:
            if rule == old_rule:  # check for isomorphism
                old_rule.frequency += 1
                return old_rule.id

        # new rule
        self.num_active_rules += 1
        rule.id = len(self.rule_list)
        self.rule_list.append(rule)
        self.rule_dict[rule.lhs].append(rule)
        return rule.id

    def calculate_cost(self):
        for rule in self.rule_list:
            if rule.is_active:  # only count if rules are active
                rule.calculate_cost()
                self.cost += rule.cost