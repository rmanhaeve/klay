"""
Simplified Python implementation of the circuit stratification logic.

This creates identical layer structures to the C++ implementation but uses
plain Python dicts instead of custom hash maps.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional


def mix_hash(h: int) -> int:
    """Improve bit dispersion of a given hash value."""
    return ((h ^ (h << 16) ^ 89869747) * 3644798167) & ((1 << 64) - 1)


class NodeType(Enum):
    TRUE = auto()
    FALSE = auto()
    OR = auto()
    AND = auto()
    LEAF = auto()


# Base hashes for node types (same as C++)
BASE_HASH_AND = 13643702618494718795
BASE_HASH_OR = 10911628454825363117
BASE_HASH_TRUE = 10398838469117805359
BASE_HASH_FALSE = 2055047638380880996


@dataclass
class Node:
    type: NodeType
    ix: int = -1  # Index in layer, -1 if uninitialized
    children: list["Node"] = field(default_factory=list)
    layer: int = 0
    hash: int = 0

    def add_child(self, child: "Node") -> None:
        """Add a child to this node, updating hash and layer."""
        if self.type not in (NodeType.OR, NodeType.AND):
            raise RuntimeError("Can only add children to AND/OR nodes")

        self.children.append(child)
        self.hash ^= mix_hash(child.hash)

        layer_bound = child.layer + 1
        # AND nodes must be in odd layers, OR nodes in even layers
        if layer_bound % 2 == 0 and self.type == NodeType.AND:
            layer_bound += 1
        elif layer_bound % 2 == 1 and self.type == NodeType.OR:
            layer_bound += 1
        self.layer = max(self.layer, layer_bound)

    def dummy_parent(self) -> "Node":
        """Create a dummy parent one layer above this node."""
        dummy = create_and_node() if self.layer % 2 == 0 else create_or_node()
        dummy.add_child(self)
        return dummy

    @property
    def is_true(self) -> bool:
        return self.type == NodeType.TRUE

    @property
    def is_false(self) -> bool:
        return self.type == NodeType.FALSE

    def __repr__(self) -> str:
        type_char = {"TRUE": "T", "FALSE": "F", "OR": "O", "AND": "A", "LEAF": "L"}[
            self.type.name
        ]
        return f"{type_char}{self.layer}/{self.ix}"


def create_literal_node(lit: int) -> Node:
    """Create a literal node. lit != 0, negative means negated."""
    if lit == 0:
        raise ValueError("Literal cannot be 0")
    return Node(type=NodeType.LEAF, ix=lit, layer=0, hash=mix_hash(lit))


def create_and_node() -> Node:
    return Node(type=NodeType.AND, ix=-1, layer=0, hash=BASE_HASH_AND)


def create_or_node() -> Node:
    return Node(type=NodeType.OR, ix=-1, layer=0, hash=BASE_HASH_OR)


def create_true_node() -> Node:
    return Node(type=NodeType.TRUE, ix=1, layer=0, hash=BASE_HASH_TRUE)


def create_false_node() -> Node:
    return Node(type=NodeType.FALSE, ix=0, layer=0, hash=BASE_HASH_FALSE)


class Circuit:
    """
    A stratified circuit where nodes are organized into layers.
    OR nodes live in even layers, AND nodes in odd layers.
    Uses dicts keyed by (hash, layer) for deduplication.
    """

    def __init__(self):
        # layers[i] is a dict: (hash, layer) -> Node
        self.layers: list[dict[tuple[int, int], Node]] = []
        self.roots: list[Node] = []

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            self.layers.append({})

    def _add_node(self, node: Node) -> Node:
        """
        Add node to circuit. Returns existing node if equivalent one exists.
        """
        self._ensure_layer(node.layer)
        layer = self.layers[node.layer]
        key = (node.hash, node.layer)

        if key in layer:
            return layer[key]

        if node.ix == -1:
            node.ix = len(layer)
        layer[key] = node
        return node

    def add_node_level(self, node: Node) -> Node:
        """
        Add node ensuring each child is in the previous adjacent layer.
        Adds dummy nodes to bridge gaps.
        """
        new_children = []
        for child in node.children:
            while child.layer < node.layer - 1:
                child = self._add_node(child.dummy_parent())
            new_children.append(child)
        node.children = new_children
        return self._add_node(node)

    def add_node_level_compressed(self, node: Node) -> Node:
        """
        Add node with compression: removes neutral elements, short-circuits on
        annihilating elements.
        """
        if node.type not in (NodeType.AND, NodeType.OR):
            return self.add_node_level(node)

        if node.type == NodeType.OR:
            annihilate_type = NodeType.TRUE
            neutral_type = NodeType.FALSE
            annihilate_fn = create_true_node
            neutral_fn = create_false_node
        else:  # AND
            annihilate_type = NodeType.FALSE
            neutral_type = NodeType.TRUE
            annihilate_fn = create_false_node
            neutral_fn = create_true_node

        new_children = []
        for child in node.children:
            if child.type == neutral_type:
                continue
            elif child.type == annihilate_type:
                return self.add_node_level(annihilate_fn())
            else:
                new_children.append(child)

        if not new_children:
            return self.add_node_level(neutral_fn())

        if len(new_children) == 1:
            return new_children[0]

        if len(new_children) != len(node.children):
            # Recreate node with correct hash
            new_node = create_and_node() if node.type == NodeType.AND else create_or_node()
            for child in new_children:
                new_node.add_child(child)
            node = new_node

        return self.add_node_level(node)

    def set_root(self, node: Node) -> None:
        self.roots.append(node)

    def add_root_layer(self) -> None:
        """Move all roots to a new top layer."""
        if not self.roots:
            raise RuntimeError("Cannot construct root layer, there are no roots!")

        root_layer_index = len(self.layers)
        new_roots = []
        for i, root in enumerate(self.roots):
            while root.layer < root_layer_index:
                root = root.dummy_parent()
                if root.layer == root_layer_index:
                    root.hash = i  # Order roots by their index
                root = self._add_node(root)
            new_roots.append(root)
        self.roots = new_roots

    def remove_unused_nodes(self) -> None:
        """Remove nodes not reachable from roots."""
        if len(self.layers) <= 1:
            return

        used: list[set[int]] = [set() for _ in range(len(self.layers))]

        # Mark roots as used
        for root in self.roots:
            if root.layer != 0:
                used[root.layer].add(root.ix)

        # Backward pass: mark children of used nodes
        for layer_idx in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[layer_idx]
            for node in layer.values():
                if node.layer == 0:
                    continue
                if node.ix in used[node.layer]:
                    for child in node.children:
                        if child.layer != 0:
                            used[child.layer].add(child.ix)

        # Remove unused nodes (skip input layer)
        for i in range(1, len(self.layers)):
            self.layers[i] = {
                k: v for k, v in self.layers[i].items() if v.ix in used[i]
            }

        # Remove trailing empty layers
        while len(self.layers) > 1 and not self.layers[-1]:
            self.layers.pop()

        # Update indices
        for i in range(1, len(self.layers)):
            for idx, node in enumerate(self.layers[i].values()):
                node.ix = idx

    def nb_layers(self) -> int:
        return len(self.layers)

    def nb_nodes(self) -> int:
        return sum(len(layer) for layer in self.layers)

    def get_indices(self) -> tuple[list[list[int]], list[list[int]]]:
        """
        Returns (indices, csr_offsets) per layer.
        indices[i] contains child indices for layer i+1.
        csr_offsets[i] contains cumulative child counts for layer i+1.
        """
        self.remove_unused_nodes()
        self.add_root_layer()

        indices_list = []
        csr_list = []

        for i in range(1, len(self.layers)):
            layer_nodes = sorted(self.layers[i].values(), key=lambda n: n.ix)
            indices = []
            csr = [0]

            for node in layer_nodes:
                for child in node.children:
                    indices.append(child.ix)
                csr.append(len(indices))

            indices_list.append(indices)
            csr_list.append(csr)

        return indices_list, csr_list

    # Convenience methods
    def true_node(self) -> Node:
        return self.add_node_level_compressed(create_true_node())

    def false_node(self) -> Node:
        return self.add_node_level_compressed(create_false_node())

    def literal_node(self, lit: int) -> Node:
        return self.add_node_level_compressed(create_literal_node(lit))

    def and_node(self, children: list[Node]) -> Node:
        node = create_and_node()
        for child in children:
            node.add_child(child)
        return self.add_node_level_compressed(node)

    def or_node(self, children: list[Node]) -> Node:
        node = create_or_node()
        for child in children:
            node.add_child(child)
        return self.add_node_level_compressed(node)

    def print_circuit(self) -> None:
        """Print circuit for debugging."""
        for i, layer in enumerate(self.layers):
            print(f"--- Layer {i} ---")
            for node in layer.values():
                children_str = ", ".join(repr(c) for c in node.children)
                print(f"  {node!r} -> [{children_str}]")

    def add_d4_from_file(
        self,
        filename: str,
        true_lits: list[int] | None = None,
        false_lits: list[int] | None = None,
    ) -> Node:
        """
        Parse a D4 NNF file and add nodes to the circuit.
        Returns the root node.
        """
        true_lits = true_lits or []
        false_lits = false_lits or []

        nodes: list[Node | None] = [None]  # 1-indexed

        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                first_char = line[0]
                if first_char == "o":
                    nodes.append(create_or_node())
                elif first_char == "a":
                    nodes.append(create_and_node())
                elif first_char == "f":
                    nodes.append(create_false_node())
                elif first_char == "t":
                    nodes.append(create_true_node())
                else:
                    # Parse edge: parent child lit1 lit2 ... 0
                    parts = list(map(int, line.split()))
                    parent_idx, child_idx = parts[0], parts[1]
                    lits = parts[2:]

                    # Finalize child when used
                    nodes[child_idx] = self.add_node_level_compressed(nodes[child_idx])
                    child = nodes[child_idx]

                    if lits == [0]:
                        # Pure edge with no literals
                        nodes[parent_idx].add_child(child)
                    else:
                        # Edge with literals
                        parent = nodes[parent_idx]
                        if parent.type == NodeType.AND:
                            edge = parent
                        else:
                            edge = create_and_node()

                        edge.add_child(child)
                        for lit in lits:
                            if lit == 0:
                                break
                            if lit in true_lits:
                                lit_node = create_true_node()
                            elif lit in false_lits:
                                lit_node = create_false_node()
                            else:
                                lit_node = create_literal_node(lit)
                            edge.add_child(self.add_node_level_compressed(lit_node))

                        if edge is not parent:
                            edge = self.add_node_level_compressed(edge)
                            parent.add_child(edge)

        # Root node (index 1) is never used as child, finalize manually
        nodes[1] = self.add_node_level_compressed(nodes[1])
        root = nodes[1]
        self.roots.append(root)
        return root

    def add_sdd_from_file(
        self,
        filename: str,
        true_lits: list[int] | None = None,
        false_lits: list[int] | None = None,
    ) -> Node:
        """
        Parse an SDD file and add nodes to the circuit.
        Returns the root node.
        """
        true_lits = true_lits or []
        false_lits = false_lits or []

        node_ids: list[Node | None] = []

        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("c"):
                    continue

                parts = line.split()
                node_type = parts[0]

                if node_type == "sdd":
                    nb_nodes = int(parts[1])
                    node_ids = [None] * nb_nodes
                    continue

                node_id = int(parts[1])

                if node_type == "F":
                    node = create_false_node()
                elif node_type == "T":
                    node = create_true_node()
                elif node_type == "L":
                    # L id vtree literal
                    literal = int(parts[3])
                    if literal in true_lits:
                        node = create_true_node()
                    elif literal in false_lits:
                        node = create_false_node()
                    else:
                        node = create_literal_node(literal)
                elif node_type == "D":
                    # D id vtree num_elements {prime_id sub_id}*
                    num_elements = int(parts[3])
                    node = create_or_node()
                    for i in range(num_elements):
                        prime_id = int(parts[4 + 2 * i])
                        sub_id = int(parts[5 + 2 * i])
                        and_node = create_and_node()
                        and_node.add_child(node_ids[prime_id])
                        and_node.add_child(node_ids[sub_id])
                        and_node = self.add_node_level_compressed(and_node)
                        node.add_child(and_node)
                else:
                    raise ValueError(f"Unknown node type: {node_type}")

                node = self.add_node_level_compressed(node)
                node_ids[node_id] = node

        # Last parsed node is the root (node_id 0 is typically last in SDD files)
        root = node
        self.roots.append(root)
        return root

    def layer_sizes(self) -> list[int]:
        """Return number of nodes per layer."""
        return [len(layer) for layer in self.layers]


if __name__ == "__main__":
    # Simple test
    c = Circuit()
    a = c.literal_node(1)
    b = c.literal_node(2)
    and1 = c.and_node([a, b])
    c.set_root(and1)
    c.print_circuit()
    print(f"\nNodes: {c.nb_nodes()}, Layers: {c.nb_layers()}")
