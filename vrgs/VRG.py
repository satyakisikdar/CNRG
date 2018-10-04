from collections import defaultdict

class VRG:
    """
    Class for Vertex Replacement Grammars
    """
    def __init__(self, mode='full', k=0):
        self.mode = mode  # type of VRG - full, part, or no
        self.rule_list = []   # list of rule objects

        self.rule_dict = defaultdict(list)  # dictionary of rules, keyed in by their LHS
        self.mdl = 0  # the MDL of the rules
        self.k = k

    def __len__(self):
        return len(self.rule_list)

    def __contains__(self, item):
        return item in self.rule_dict[item.lhs]


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

    def calculate_cost(self, contract):
        for rule in self.rule_list:
            if contract:
                rule.contract_rhs()
            rule.calculate_cost()
            self.mdl += rule.cost

    def get_cost(self, contract=False):
        # if self.mdl != 0:  # the cost has been computed before
        #     return self.mdl
        # else:
        self.mdl = 0
        self.calculate_cost(contract)
        return self.mdl