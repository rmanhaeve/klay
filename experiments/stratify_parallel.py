"""
Parallel-rank stratification: AND and OR nodes can share the same rank.

Unlike the alternating approach, nodes are placed at rank = max(child ranks) + 1
without parity constraints. Skip/pass-through nodes enforce strict adjacency
(children always at rank-1).

Each rank has separate AND and OR node sets, evaluated in parallel.
"""

from enum import Enum, auto
from dataclasses import dataclass, field


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
    rank: int = 0  # Rank (not strictly alternating)
    hash: int = 0

    def add_child(self, child: "Node") -> None:
        """Add a child to this node, updating hash and rank."""
        if self.type not in (NodeType.OR, NodeType.AND):
            raise RuntimeError("Can only add children to AND/OR nodes")

        self.children.append(child)
        self.hash ^= mix_hash(child.hash)
        # No parity constraint — just go one above highest child
        self.rank = max(self.rank, child.rank + 1)

    def skip_parent(self) -> "Node":
        """Create a dummy parent one rank above this node (single-child AND or OR)."""
        # Alternate AND/OR like the alternating approach does
        if self.rank % 2 == 0:
            dummy = create_and_node()
        else:
            dummy = create_or_node()
        dummy.add_child(self)
        return dummy

    @property
    def is_true(self) -> bool:
        return self.type == NodeType.TRUE

    @property
    def is_false(self) -> bool:
        return self.type == NodeType.FALSE

    def __repr__(self) -> str:
        type_char = {
            "TRUE": "T", "FALSE": "F", "OR": "O", 
            "AND": "A", "LEAF": "L"
        }[self.type.name]
        return f"{type_char}{self.rank}/{self.ix}"


def create_literal_node(lit: int) -> Node:
    if lit == 0:
        raise ValueError("Literal cannot be 0")
    return Node(type=NodeType.LEAF, ix=lit, rank=0, hash=mix_hash(lit))


def create_and_node() -> Node:
    return Node(type=NodeType.AND, ix=-1, rank=0, hash=BASE_HASH_AND)


def create_or_node() -> Node:
    return Node(type=NodeType.OR, ix=-1, rank=0, hash=BASE_HASH_OR)


def create_true_node() -> Node:
    return Node(type=NodeType.TRUE, ix=1, rank=0, hash=BASE_HASH_TRUE)


def create_false_node() -> Node:
    return Node(type=NodeType.FALSE, ix=0, rank=0, hash=BASE_HASH_FALSE)


class ParallelCircuit:
    """
    Parallel-rank circuit: AND and OR nodes can coexist at the same rank.
    Uses dummy AND/OR nodes (like alternating) to enforce strict adjacency.
    """

    def __init__(self):
        # Parallel lists by rank
        self.input_layer: dict[tuple[int, int], Node] = {}
        self.and_layers: list[dict[tuple[int, int], Node]] = []
        self.or_layers: list[dict[tuple[int, int], Node]] = []
        self.roots: list[Node] = []

    def _ensure_rank(self, rank: int) -> None:
        while len(self.and_layers) < rank:
            self.and_layers.append({})
            self.or_layers.append({})

    def _get_layer_for_type(self, node: Node) -> dict[tuple[int, int], Node]:
        """Get the appropriate layer dict for a node."""
        if node.rank == 0:
            return self.input_layer
        self._ensure_rank(node.rank)
        if node.type == NodeType.AND:
            return self.and_layers[node.rank - 1]
        elif node.type == NodeType.OR:
            return self.or_layers[node.rank - 1]
        else:
            raise RuntimeError(f"Unexpected node type {node.type} at rank {node.rank}")

    def _add_node(self, node: Node) -> Node:
        """Add node to circuit. Returns existing node if equivalent one exists."""
        layer = self._get_layer_for_type(node)
        key = (node.hash, node.rank)

        if key in layer:
            return layer[key]

        if node.ix == -1:
            node.ix = len(layer)
        layer[key] = node
        return node

    def add_node_ranked(self, node: Node) -> Node:
        """
        Add node ensuring each child is at rank-1 (strict adjacency).
        Adds skip nodes to bridge gaps.
        """
        new_children = []
        for child in node.children:
            while child.rank < node.rank - 1:
                child = self._add_node(child.skip_parent())
            new_children.append(child)
        node.children = new_children
        return self._add_node(node)

    def add_node_compressed(self, node: Node) -> Node:
        """Add node with compression (remove neutral, short-circuit annihilating)."""
        if node.type not in (NodeType.AND, NodeType.OR):
            return self.add_node_ranked(node)

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
                return self.add_node_ranked(annihilate_fn())
            else:
                new_children.append(child)

        if not new_children:
            return self.add_node_ranked(neutral_fn())

        if len(new_children) == 1:
            return new_children[0]

        if len(new_children) != len(node.children):
            new_node = create_and_node() if node.type == NodeType.AND else create_or_node()
            for child in new_children:
                new_node.add_child(child)
            node = new_node

        return self.add_node_ranked(node)

    def set_root(self, node: Node) -> None:
        self.roots.append(node)

    def nb_ranks(self) -> int:
        """Number of ranks (including input rank 0)."""
        return len(self.and_layers) + 1

    def nb_nodes(self) -> int:
        """Total number of nodes."""
        count = len(self.input_layer)
        for i in range(len(self.and_layers)):
            count += len(self.and_layers[i])
            count += len(self.or_layers[i])
        return count

    def rank_sizes(self) -> list[tuple[int, int]]:
        """Return (and_count, or_count) per rank (excluding input)."""
        sizes = []
        for i in range(len(self.and_layers)):
            sizes.append((
                len(self.and_layers[i]),
                len(self.or_layers[i]),
            ))
        return sizes

    def add_root_rank(self) -> None:
        """Move all roots to a new top rank."""
        if not self.roots:
            raise RuntimeError("No roots!")

        root_rank = len(self.and_layers) + 1
        new_roots = []
        for i, root in enumerate(self.roots):
            while root.rank < root_rank:
                root = root.skip_parent()
                if root.rank == root_rank:
                    root.hash = i
                root = self._add_node(root)
            new_roots.append(root)
        self.roots = new_roots

    def remove_unused_nodes(self) -> None:
        """Remove nodes not reachable from roots."""
        if not self.and_layers:
            return

        # Track used nodes by (rank, type, ix)
        used: dict[int, set[tuple[str, int]]] = {r: set() for r in range(self.nb_ranks())}

        # Mark roots
        for root in self.roots:
            if root.rank > 0:
                used[root.rank].add((root.type.name, root.ix))

        # Backward pass
        for rank in range(len(self.and_layers), 0, -1):
            for layer, type_name in [
                (self.and_layers[rank - 1], "AND"),
                (self.or_layers[rank - 1], "OR"),
            ]:
                for node in layer.values():
                    if (type_name, node.ix) in used[rank]:
                        for child in node.children:
                            if child.rank > 0:
                                used[child.rank].add((child.type.name, child.ix))

        # Remove unused
        for rank in range(1, len(self.and_layers) + 1):
            for layer, type_name in [
                (self.and_layers[rank - 1], "AND"),
                (self.or_layers[rank - 1], "OR"),
            ]:
                to_remove = [k for k, v in layer.items() if (type_name, v.ix) not in used[rank]]
                for k in to_remove:
                    del layer[k]

        # Remove trailing empty ranks
        while self.and_layers and not (self.and_layers[-1] or self.or_layers[-1]):
            self.and_layers.pop()
            self.or_layers.pop()

        # Update indices
        for rank in range(1, len(self.and_layers) + 1):
            for layer in [self.and_layers[rank - 1], self.or_layers[rank - 1]]:
                for idx, node in enumerate(layer.values()):
                    node.ix = idx

    def get_indices(self) -> dict:
        """
        Returns indices split by node type per rank.
        {
            'and': [(indices, csr), ...],  # per rank
            'or': [(indices, csr), ...],
        }
        """
        self.remove_unused_nodes()
        self.add_root_rank()

        def build_indices(layers):
            result = []
            for layer in layers:
                nodes = sorted(layer.values(), key=lambda n: n.ix)
                indices = []
                csr = [0]
                for node in nodes:
                    for child in node.children:
                        indices.append(child.ix)
                    csr.append(len(indices))
                result.append((indices, csr))
            return result

        return {
            'and': build_indices(self.and_layers),
            'or': build_indices(self.or_layers),
        }

    # Convenience methods
    def true_node(self) -> Node:
        return self.add_node_compressed(create_true_node())

    def false_node(self) -> Node:
        return self.add_node_compressed(create_false_node())

    def literal_node(self, lit: int) -> Node:
        return self.add_node_compressed(create_literal_node(lit))

    def and_node(self, children: list[Node]) -> Node:
        node = create_and_node()
        for child in children:
            node.add_child(child)
        return self.add_node_compressed(node)

    def or_node(self, children: list[Node]) -> Node:
        node = create_or_node()
        for child in children:
            node.add_child(child)
        return self.add_node_compressed(node)

    def add_d4_from_file(
        self,
        filename: str,
        true_lits: list[int] | None = None,
        false_lits: list[int] | None = None,
    ) -> Node:
        """Parse a D4 NNF file and add nodes to the circuit."""
        true_lits = true_lits or []
        false_lits = false_lits or []

        nodes: list[Node | None] = [None]

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
                    parts = list(map(int, line.split()))
                    parent_idx, child_idx = parts[0], parts[1]
                    lits = parts[2:]

                    nodes[child_idx] = self.add_node_compressed(nodes[child_idx])
                    child = nodes[child_idx]

                    if lits == [0]:
                        nodes[parent_idx].add_child(child)
                    else:
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
                            edge.add_child(self.add_node_compressed(lit_node))

                        if edge is not parent:
                            edge = self.add_node_compressed(edge)
                            parent.add_child(edge)

        nodes[1] = self.add_node_compressed(nodes[1])
        root = nodes[1]
        self.roots.append(root)
        return root

    def add_sdd_from_file(
        self,
        filename: str,
        true_lits: list[int] | None = None,
        false_lits: list[int] | None = None,
    ) -> Node:
        """Parse an SDD file and add nodes to the circuit."""
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
                    literal = int(parts[3])
                    if literal in true_lits:
                        node = create_true_node()
                    elif literal in false_lits:
                        node = create_false_node()
                    else:
                        node = create_literal_node(literal)
                elif node_type == "D":
                    num_elements = int(parts[3])
                    node = create_or_node()
                    for i in range(num_elements):
                        prime_id = int(parts[4 + 2 * i])
                        sub_id = int(parts[5 + 2 * i])
                        and_node = create_and_node()
                        and_node.add_child(node_ids[prime_id])
                        and_node.add_child(node_ids[sub_id])
                        and_node = self.add_node_compressed(and_node)
                        node.add_child(and_node)
                else:
                    raise ValueError(f"Unknown node type: {node_type}")

                node = self.add_node_compressed(node)
                node_ids[node_id] = node

        # Last parsed node is the root (node_id 0 is typically last in SDD files)
        root = node
        self.roots.append(root)
        return root

    def print_circuit(self) -> None:
        """Print circuit for debugging."""
        print(f"--- Input (rank 0) ---")
        for node in self.input_layer.values():
            print(f"  {node!r}")
        
        for r in range(len(self.and_layers)):
            print(f"--- Rank {r + 1} ---")
            for node in self.and_layers[r].values():
                children_str = ", ".join(repr(c) for c in node.children)
                print(f"  {node!r} -> [{children_str}]")
            for node in self.or_layers[r].values():
                children_str = ", ".join(repr(c) for c in node.children)
                print(f"  {node!r} -> [{children_str}]")


if __name__ == "__main__":
    # Simple test
    c = ParallelCircuit()
    a = c.literal_node(1)
    b = c.literal_node(2)
    and1 = c.and_node([a, b])
    or1 = c.or_node([a, b])
    top = c.and_node([and1, or1])
    c.set_root(top)
    c.print_circuit()
    print(f"\nNodes: {c.nb_nodes()}, Ranks: {c.nb_ranks()}")
    print(f"Rank sizes (and, or, skip): {c.rank_sizes()}")
