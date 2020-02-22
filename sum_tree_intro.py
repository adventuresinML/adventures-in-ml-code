import numpy as np

class Node:
    def __init__(self, left, right, is_leaf: bool = False, idx = None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
        self.parent = None
        self.idx = idx  # this value is only set for leaf nodes
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf


def create_tree(input: list):
    nodes = [Node.create_leaf(v, i) for i, v in enumerate(input)]
    leaf_nodes = nodes
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]

    return nodes[0], leaf_nodes


def retrieve(value: float, node: Node):
    if node.is_leaf:
        return node

    if node.left.value >= value:
        return retrieve(value, node.left)
    else:
        return retrieve(value - node.left.value, node.right)


def update(node: Node, new_value: float):
    change = new_value - node.value

    node.value = new_value
    propagate_changes(change, node.parent)


def propagate_changes(change: float, node: Node):
    node.value += change

    if node.parent is not None:
        propagate_changes(change, node.parent)


def demonstrate_sampling(root_node: Node):
    tree_total = root_node.value
    iterations = 1000000
    selected_vals = []
    for i in range(iterations):
        rand_val = np.random.uniform(0, tree_total)
        selected_val = retrieve(rand_val, root_node).value
        selected_vals.append(selected_val)
    
    return selected_vals

input = [1, 4, 2, 3]

root_node, leaf_nodes = create_tree(input)
selected_vals = demonstrate_sampling(root_node)
# the below print statement should output ~4
print(f"Should be ~4: {sum([1 for x in selected_vals if x == 4]) / sum([1 for y in selected_vals if y == 1])}")

update(leaf_nodes[1], 6)
selected_vals = demonstrate_sampling(root_node)
# the below print statement should output ~6
print(f"Should be ~6: {sum([1 for x in selected_vals if x == 6]) / sum([1 for y in selected_vals if y == 1])}")
# the below print statement should output ~2
print(f"Should be ~2: {sum([1 for x in selected_vals if x == 6]) / sum([1 for y in selected_vals if y == 3])}")



