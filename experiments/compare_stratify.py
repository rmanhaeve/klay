"""
Compare Python stratification with C++ klay implementation.

Usage:
    python compare_stratify.py <nnf_file>   # Compare on a specific NNF file
    python compare_stratify.py              # Generate random CNF and compare (requires d4 compiler)
"""

import sys
from pathlib import Path

from stratify import Circuit as PyCircuit

try:
    import klay
    KLAY_AVAILABLE = True
except ImportError:
    KLAY_AVAILABLE = False
    print("Warning: klay C++ module not available, will only show Python results")


def compare_d4_file(nnf_file: str):
    """Compare layer counts between C++ and Python on a D4 NNF file."""
    print(f"=== Comparing on {nnf_file} ===")

    # Python implementation
    py_circuit = PyCircuit()
    py_circuit.add_d4_from_file(nnf_file)
    py_nodes_before = py_circuit.nb_nodes()
    py_layers_before = py_circuit.nb_layers()
    py_layer_sizes_before = py_circuit.layer_sizes()

    py_indices, py_csr = py_circuit.get_indices()
    py_layers = len(py_indices) + 1
    py_layer_sizes = [len(csr) - 1 for csr in py_csr]

    print(f"Py  nodes (before get_indices): {py_nodes_before}")
    print(f"Py  layers (before): {py_layers_before}, sizes: {py_layer_sizes_before}")
    print(f"Py  layers (after):  {py_layers}, sizes: {py_layer_sizes}")

    if KLAY_AVAILABLE:
        # C++ implementation
        cpp_circuit = klay.Circuit()
        cpp_circuit.add_d4_from_file(nnf_file)
        cpp_nodes = cpp_circuit.nb_nodes()

        cpp_indices, cpp_csr = cpp_circuit._get_indices()
        cpp_layers = len(cpp_indices) + 1
        cpp_layer_sizes = [len(csr) - 1 for csr in cpp_csr]

        print(f"C++ nodes: {cpp_nodes}")
        print(f"C++ layers: {cpp_layers}, sizes: {cpp_layer_sizes}")

        match = (cpp_layers == py_layers) and (cpp_layer_sizes == py_layer_sizes)
        print(f"Match: {'✓' if match else '✗'}")
        return match

    return True


def compare_random(nb_vars: int, seed: int = 0):
    """Generate random CNF, compile with D4, and compare."""
    from klay.utils import generate_random_dimacs
    from klay.compile import compile_d4

    cnf_file = "/tmp/compare_test.cnf"
    nnf_file = "/tmp/compare_test.nnf"

    generate_random_dimacs(cnf_file, nb_vars, 2 * nb_vars, seed=seed)
    compile_d4(cnf_file, nnf_file)

    return compare_d4_file(nnf_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nnf_file", nargs="?", help="D4 NNF file to compare")
    parser.add_argument("-v", "--nb_vars", type=int, default=10)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--num_tests", type=int, default=1)
    args = parser.parse_args()

    if args.nnf_file:
        success = compare_d4_file(args.nnf_file)
        sys.exit(0 if success else 1)
    else:
        all_match = True
        for seed in range(args.seed, args.seed + args.num_tests):
            print(f"\n--- Test seed={seed} ---")
            if not compare_random(args.nb_vars, seed):
                all_match = False
        sys.exit(0 if all_match else 1)
