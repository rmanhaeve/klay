"""
Compare stratification implementations.

Usage:
    python compare_stratify.py <nnf_file>   # Compare on a specific NNF file
    python compare_stratify.py              # Generate random CNF and compare (requires d4 compiler)
"""

import sys
from pathlib import Path

from stratify import Circuit as AlternatingCircuit
from stratify_parallel import ParallelCircuit

try:
    import klay
    KLAY_AVAILABLE = True
except ImportError:
    KLAY_AVAILABLE = False
    print("Warning: klay C++ module not available")


def compare_file(filename: str):
    """Compare alternating vs parallel stratification on an SDD or D4 file."""
    print(f"=== Comparing on {filename} ===\n")

    is_sdd = filename.endswith(".sdd")

    # Alternating (original) implementation
    alt_circuit = AlternatingCircuit()
    if is_sdd:
        alt_circuit.add_sdd_from_file(filename)
    else:
        alt_circuit.add_d4_from_file(filename)
    alt_nodes_before = alt_circuit.nb_nodes()
    alt_layers_before = alt_circuit.nb_layers()

    alt_indices, alt_csr = alt_circuit.get_indices()
    alt_layers = len(alt_indices) + 1
    alt_layer_sizes = [len(csr) - 1 for csr in alt_csr]

    print("ALTERNATING (original):")
    print(f"  Nodes (before finalize): {alt_nodes_before}")
    print(f"  Layers (before): {alt_layers_before}")
    print(f"  Layers (after):  {alt_layers}")
    print(f"  Max layer width: {max(alt_layer_sizes)}")
    print(f"  Layer sizes: {alt_layer_sizes[:10]}{'...' if len(alt_layer_sizes) > 10 else ''}")
    print()

    # Parallel implementation
    par_circuit = ParallelCircuit()
    if is_sdd:
        par_circuit.add_sdd_from_file(filename)
    else:
        par_circuit.add_d4_from_file(filename)
    par_nodes_before = par_circuit.nb_nodes()
    par_ranks_before = par_circuit.nb_ranks()

    par_indices = par_circuit.get_indices()
    par_ranks = par_circuit.nb_ranks()
    
    # Compute totals and max width per rank
    par_rank_totals = []
    par_max_width = 0
    for i in range(len(par_indices['and'])):
        and_count = len(par_indices['and'][i][1]) - 1
        or_count = len(par_indices['or'][i][1]) - 1
        par_rank_totals.append((and_count, or_count))
        par_max_width = max(par_max_width, and_count + or_count)

    print("PARALLEL (new):")
    print(f"  Nodes (before finalize): {par_nodes_before}")
    print(f"  Ranks (before): {par_ranks_before}")
    print(f"  Ranks (after):  {par_ranks}")
    print(f"  Max rank width: {par_max_width}")
    print(f"  Rank sizes (and, or): {par_rank_totals[:10]}{'...' if len(par_rank_totals) > 10 else ''}")
    print()

    # Summary comparison
    alt_total_ops = sum(alt_layer_sizes)
    par_total_ops = sum(a + o for a, o in par_rank_totals)
    
    print("COMPARISON:")
    print(f"  Alternating: {alt_layers} layers, {alt_total_ops} total nodes (excl. input)")
    print(f"  Parallel:    {par_ranks} ranks, {par_total_ops} total nodes (excl. input)")
    print(f"  Rank reduction: {alt_layers} -> {par_ranks} ({alt_layers - par_ranks} fewer)")
    print(f"  Node reduction: {alt_total_ops} -> {par_total_ops} ({alt_total_ops - par_total_ops} fewer)")
    
    # C++ comparison
    if KLAY_AVAILABLE:
        cpp_circuit = klay.Circuit()
        if is_sdd:
            cpp_circuit.add_sdd_from_file(filename)
        else:
            cpp_circuit.add_d4_from_file(filename)
        cpp_indices, cpp_csr = cpp_circuit._get_indices()
        cpp_layers = len(cpp_indices) + 1
        cpp_layer_sizes = [len(csr) - 1 for csr in cpp_csr]
        
        alt_match = (alt_layers == cpp_layers) and (alt_layer_sizes == cpp_layer_sizes)
        print(f"\n  Alternating matches C++: {'✓' if alt_match else '✗'}")

    return True


def compare_nesy_circuits():
    """Compare on all nesy benchmark circuits."""
    circuits = ["sudoku_4", "4-grid", "seq_fun", "warcraft_12"]
    
    results = []
    for name in circuits:
        sdd_file = f"nesy/circuits/{name}.sdd"
        print(f"\n{'='*60}")
        compare_file(sdd_file)
        results.append(name)
    
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="SDD or NNF file to compare")
    parser.add_argument("--nesy", action="store_true", help="Compare all nesy circuits")
    args = parser.parse_args()

    if args.nesy:
        compare_nesy_circuits()
    elif args.file:
        compare_file(args.file)
    else:
        parser.print_help()
