class TreeNode:
    """
    Node class for trees
    """
    def __init__(self, key, payload=set(), is_leaf=False):
        self.key = key   # key of the node, each node has an unique key
        self.payload = payload  # payload on the node, could be anything

        self.parent = None
        self.left = None  # left child
        self.right = None  # right child

        self.is_leaf = is_leaf  # True if it's a child, False otherwise
        self.nleaf = 0   # number of leaves


    def __str__(self):
        if len(self.payload) != 0:
            return '{} ({}): {}'.format(self.key, self.nleaf, self.payload)

        if self.parent is None:
            parent = None
        else:
            parent = self.parent.key

        return '{} ({}) p: {}'.format(self.key, self.nleaf, parent)

    def __repr__(self):
        return '{} ({})'.format(self.key, self.nleaf)

    def __lt__(self, other):
        return self.nleaf < other.nleaf


def create_tree(lst):
    """
    Creates a BinaryTree from the list of lists
    :param lst: nested list of lists
    :return: root of binary tree
    """
    key = 'a'

    def create(lst):
        nonlocal key
        if len(lst) == 1:  # detect leaf
            return TreeNode(key=lst[0], is_leaf=True)

        node = TreeNode(key=key)
        key = chr(ord(key) + 1)
        node.left = create(lst[0])
        node.right = create(lst[1])

        return node

    root = create(lst)

    def update_info(node):
        """
        updates the parent pointers, payloads, and the number of leaf nodes
        :param node:
        :return:
        """
        if node.is_leaf:
            node.nleaf = 1
            node.payload = {node.key}

        else:
            if node.left is not None:
                node.left.parent = node
                nleaf, pl = update_info(node.left)
                node.nleaf += nleaf
                node.payload = pl.copy()

            if node.right is not None:
                node.right.parent = node
                nleaf, pl = update_info(node.right)
                node.nleaf += nleaf
                node.payload.update(pl)

        return node.nleaf, node.payload.copy()

    update_info(root)

    return root