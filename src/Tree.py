class TreeNode:
    """
    Node class for trees
    """
    def __init__(self, key, is_leaf=False):
        self.key = key   # key of the node, each node has an unique key
        self.level = 0  # level of the node

        self.children = set()  # set of node labels of nodes in the subtree rooted at the node
        self.leaves = set()  # set of children that are leaf nodes


        self.parent = None  # pointer to paren
        self.kids = []  # pointers to the children

        self.is_leaf = is_leaf  # True if it's a child, False otherwise
        self.nleaf = 0   # number of leaves


    def __str__(self):
        if self.parent is None:
            parent = None
        else:
            parent = self.parent.key

        return '{} ({}) p: {}'.format(self.key, self.nleaf, parent)

    def __repr__(self):
        return '{} ({})'.format(self.key, self.nleaf)

    def __lt__(self, other):
        return self.nleaf < other.nleaf

    def __copy__(self):
        node_copy = TreeNode(key=self.key)

        node_copy.parent = self.parent
        node_copy.kids = self.kids

        node_copy.leaves = self.leaves
        node_copy.children = self.children

        node_copy.level = self.level
        node_copy.is_leaf = self.is_leaf
        node_copy.nleaf = self.nleaf

        return node_copy

    def __hash__(self):
        return hash(self.key)

    def copy(self):
        return self.__copy__()

    def make_leaf(self, new_key):
        """
        converts the internal tree node into a leaf
        :param new_key: new key of the node
        :return:
        """
        self.key = new_key  # the best node's key is now the key of the new_node
        self.leaves = {self.key}  # update the leaves
        self.children = set()

        self.kids = []
        self.is_leaf = True
        self.nleaf = 1


def create_tree(lst):
    """
    Creates a Tree from the list of lists
    :param lst: nested list of lists
    :return: root of the tree
    """
    key = 'a'

    def create(lst):
        nonlocal key

        if len(lst) == 1 and isinstance(lst[0], int):  # detect leaf
            return TreeNode(key=lst[0], is_leaf=True)
        node = TreeNode(key=key)
        key = chr(ord(key) + 1)

        for item in lst:
            node.kids.append(create(item))

        return node

    root = create(lst)

    def update_info(node):
        """
        updates the parent pointers, payloads, and the number of leaf nodes
        :param node:
        :return:
        """
        if node.is_leaf:
            node.make_leaf(new_key=node.key)  # the key doesn't change

        else:
            for kid in node.kids:
                kid.parent = node
                nleaf, children, leaves = update_info(kid)
                node.nleaf += nleaf
                node.children.add(kid.key)
                node.children.update(children)
                node.leaves.update(leaves)

        return node.nleaf, node.children, node.leaves

    update_info(node=root)

    return root
