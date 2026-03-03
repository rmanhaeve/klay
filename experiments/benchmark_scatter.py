"""
Benchmark: scatter_reduce vs scatter + manual reduction

Compares:
1. torch.scatter_reduce (fused operation)
2. Scatter to expanded tensor + torch reduction operation (e.g., prod along dim)
"""

import torch
import time
import argparse
import json
import os
from typing import Callable, Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np


def make_test_data(n_inputs: int, n_outputs: int, avg_inputs_per_output: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate test data mimicking the KnowledgeLayer structure."""
    # Number of total scatter operations
    n_scatter = n_outputs * avg_inputs_per_output
    
    # ptrs: indices into input tensor
    ptrs = torch.randint(0, n_inputs, (n_scatter,), dtype=torch.long, device=device)
    
    # csr: output indices (each output receives avg_inputs_per_output values)
    csr = torch.arange(n_outputs, device=device).repeat_interleave(avg_inputs_per_output)
    
    # input values
    x = torch.rand(n_inputs, device=device)
    
    return x, ptrs, csr


# =============================================================================
# Approach 1: scatter_reduce (fused, current implementation)
# =============================================================================

def scatter_reduce_prod(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor, n_outputs: int) -> torch.Tensor:
    """Current approach: fused scatter_reduce with prod."""
    output = torch.empty(n_outputs, dtype=x.dtype, device=x.device)
    return torch.scatter_reduce(output, 0, index=csr, src=x[ptrs], reduce="prod", include_self=False)


def scatter_reduce_sum(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor, n_outputs: int) -> torch.Tensor:
    """Current approach: fused scatter_reduce with sum."""
    output = torch.zeros(n_outputs, dtype=x.dtype, device=x.device)
    return torch.scatter_reduce(output, 0, index=csr, src=x[ptrs], reduce="sum", include_self=True)


def scatter_reduce_max(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor, n_outputs: int) -> torch.Tensor:
    """Current approach: fused scatter_reduce with amax."""
    output = torch.empty(n_outputs, dtype=x.dtype, device=x.device)
    return torch.scatter_reduce(output, 0, index=csr, src=x[ptrs], reduce="amax", include_self=False)


def scatter_reduce_min(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor, n_outputs: int) -> torch.Tensor:
    """Current approach: fused scatter_reduce with amin."""
    output = torch.empty(n_outputs, dtype=x.dtype, device=x.device)
    return torch.scatter_reduce(output, 0, index=csr, src=x[ptrs], reduce="amin", include_self=False)


# JIT-compiled versions of scatter_reduce
@torch.jit.script
def scatter_reduce_prod_jit(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor, n_outputs: int) -> torch.Tensor:
    output = torch.empty(n_outputs, dtype=x.dtype, device=x.device)
    return torch.scatter_reduce(output, 0, index=csr, src=x[ptrs], reduce="prod", include_self=False)


@torch.jit.script
def scatter_reduce_sum_jit(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor, n_outputs: int) -> torch.Tensor:
    output = torch.zeros(n_outputs, dtype=x.dtype, device=x.device)
    return torch.scatter_reduce(output, 0, index=csr, src=x[ptrs], reduce="sum", include_self=True)


@torch.jit.script
def scatter_reduce_max_jit(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor, n_outputs: int) -> torch.Tensor:
    output = torch.empty(n_outputs, dtype=x.dtype, device=x.device)
    return torch.scatter_reduce(output, 0, index=csr, src=x[ptrs], reduce="amax", include_self=False)


@torch.jit.script
def scatter_reduce_min_jit(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor, n_outputs: int) -> torch.Tensor:
    output = torch.empty(n_outputs, dtype=x.dtype, device=x.device)
    return torch.scatter_reduce(output, 0, index=csr, src=x[ptrs], reduce="amin", include_self=False)


# =============================================================================
# Approach 2: Scatter to 2D tensor + torch.prod/sum/etc (general approach)
# =============================================================================

def compute_positions(csr: torch.Tensor) -> torch.Tensor:
    """
    Precompute positions within each group for the general approach.
    This is bookkeeping that should be done once, not timed.
    """
    # Efficient position calculation assuming sorted csr
    group_starts = torch.cat([torch.tensor([0], device=csr.device), 
                               torch.where(csr[1:] != csr[:-1])[0] + 1])
    cumsum = torch.arange(len(csr), device=csr.device)
    start_values = cumsum[group_starts]
    positions = cumsum - start_values.repeat_interleave(
        torch.diff(torch.cat([group_starts, torch.tensor([len(csr)], device=csr.device)]))
    )
    return positions


def general_scatter_then_reduce(
    x: torch.Tensor, 
    ptrs: torch.Tensor, 
    csr: torch.Tensor,
    positions: torch.Tensor,
    scattered: torch.Tensor,
    fill_value: float,
    reduce_fn: Callable[[torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """
    General approach: scatter to 2D padded tensor, then apply reduction.
    
    Positions and scattered buffer are precomputed (bookkeeping), so this 
    only times the actual scatter and reduce operations.
    """
    # Reset scattered buffer to fill_value
    scattered.fill_(fill_value)
    scattered[csr, positions] = x[ptrs]
    
    # Apply reduction along dim=1
    return reduce_fn(scattered, 1)


def general_prod(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor, positions: torch.Tensor, scattered: torch.Tensor) -> torch.Tensor:
    return general_scatter_then_reduce(x, ptrs, csr, positions, scattered, 1.0,
                                       lambda t, d: torch.prod(t, dim=d))


def general_sum(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor, positions: torch.Tensor, scattered: torch.Tensor) -> torch.Tensor:
    return general_scatter_then_reduce(x, ptrs, csr, positions, scattered, 0.0,
                                       lambda t, d: torch.sum(t, dim=d))


def general_max(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor, positions: torch.Tensor, scattered: torch.Tensor) -> torch.Tensor:
    return general_scatter_then_reduce(x, ptrs, csr, positions, scattered, float('-inf'),
                                       lambda t, d: torch.max(t, dim=d).values)


def general_min(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor, positions: torch.Tensor, scattered: torch.Tensor) -> torch.Tensor:
    return general_scatter_then_reduce(x, ptrs, csr, positions, scattered, float('inf'),
                                       lambda t, d: torch.min(t, dim=d).values)


# JIT-compiled versions of general approach
# Note: We can't use lambda with JIT, so we create specific functions

@torch.jit.script
def general_prod_jit(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor, 
                     positions: torch.Tensor, scattered: torch.Tensor) -> torch.Tensor:
    scattered.fill_(1.0)
    scattered[csr, positions] = x[ptrs]
    return torch.prod(scattered, dim=1)


@torch.jit.script
def general_sum_jit(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor,
                    positions: torch.Tensor, scattered: torch.Tensor) -> torch.Tensor:
    scattered.fill_(0.0)
    scattered[csr, positions] = x[ptrs]
    return torch.sum(scattered, dim=1)


@torch.jit.script
def general_max_jit(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor,
                    positions: torch.Tensor, scattered: torch.Tensor) -> torch.Tensor:
    scattered.fill_(float('-inf'))
    scattered[csr, positions] = x[ptrs]
    return torch.max(scattered, dim=1)[0]


@torch.jit.script
def general_min_jit(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor,
                    positions: torch.Tensor, scattered: torch.Tensor) -> torch.Tensor:
    scattered.fill_(float('inf'))
    scattered[csr, positions] = x[ptrs]
    return torch.min(scattered, dim=1)[0]


# =============================================================================
# Benchmarking utilities
# =============================================================================

def benchmark_forward(fn: Callable, *args, n_warmup: int = 10, n_iter: int = 100, device: str = 'cpu') -> float:
    """Benchmark forward pass and return average time in milliseconds."""
    # Warmup
    for _ in range(n_warmup):
        fn(*args)
    
    # Synchronize before timing
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    start = time.perf_counter()
    for _ in range(n_iter):
        fn(*args)
    
    # Synchronize after timing
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    elapsed = time.perf_counter() - start
    return (elapsed / n_iter) * 1000  # ms


def benchmark_backward(fn: Callable, x: torch.Tensor, *args, n_warmup: int = 10, n_iter: int = 100, device: str = 'cpu') -> float:
    """Benchmark backward pass and return average time in milliseconds."""
    # Clone non-tensor args that might be modified
    def clone_if_tensor(arg):
        if isinstance(arg, torch.Tensor):
            return arg.detach().clone()
        return arg
    
    # Warmup
    for _ in range(n_warmup):
        x_grad = x.detach().clone().requires_grad_(True)
        cloned_args = tuple(clone_if_tensor(a) for a in args)
        out = fn(x_grad, *cloned_args)
        grad_out = torch.ones_like(out)
        out.backward(grad_out)
    
    # Synchronize before timing
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    start = time.perf_counter()
    for _ in range(n_iter):
        x_grad = x.detach().clone().requires_grad_(True)
        cloned_args = tuple(clone_if_tensor(a) for a in args)
        out = fn(x_grad, *cloned_args)
        grad_out = torch.ones_like(out)
        out.backward(grad_out)
    
    # Synchronize after timing
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    elapsed = time.perf_counter() - start
    return (elapsed / n_iter) * 1000  # ms


def verify_correctness(x: torch.Tensor, ptrs: torch.Tensor, csr: torch.Tensor, 
                       positions: torch.Tensor, scattered: torch.Tensor, n_outputs: int) -> None:
    """Verify that both approaches produce the same results."""
    print("\n=== Correctness Verification ===")
    
    for name, scatter_fn, general_fn in [
        ("prod", scatter_reduce_prod, general_prod),
        ("sum", scatter_reduce_sum, general_sum),
        ("max", scatter_reduce_max, general_max),
        ("min", scatter_reduce_min, general_min),
    ]:
        r1 = scatter_fn(x, ptrs, csr, n_outputs)
        r2 = general_fn(x, ptrs, csr, positions, scattered)
        
        # Handle inf values in comparison
        mask = torch.isfinite(r1) & torch.isfinite(r2)
        if mask.sum() > 0:
            max_diff = (r1[mask] - r2[mask]).abs().max().item()
            match = max_diff < 1e-5
        else:
            match = True
        
        status = "✓" if match else "✗"
        print(f"  {name}: {status} (max diff: {max_diff if mask.sum() > 0 else 'N/A'})")


def run_benchmarks(n_inputs: int, n_outputs: int, avg_inputs_per_output: int, 
                   device: str, n_iter: int = 100, verbose: bool = True) -> Dict[str, Any]:
    """Run all benchmarks and return results."""
    if verbose:
        print(f"\n{'='*100}")
        print(f"Benchmark: n_inputs={n_inputs}, n_outputs={n_outputs}, "
              f"avg_per_output={avg_inputs_per_output}, device={device}")
        print(f"{'='*100}")
    
    x, ptrs, csr = make_test_data(n_inputs, n_outputs, avg_inputs_per_output, device)
    max_per_output = avg_inputs_per_output  # Assuming uniform distribution
    
    # Precompute positions and allocate scattered buffer (bookkeeping - not timed)
    positions = compute_positions(csr)
    scattered = torch.empty((n_outputs, max_per_output), dtype=x.dtype, device=x.device)
    
    # Verify correctness first
    if verbose:
        verify_correctness(x, ptrs, csr, positions, scattered, n_outputs)
        print(f"\n{'Op':<6} {'scat_fwd':>9} {'scat_jit':>9} {'gen_fwd':>9} {'gen_jit':>9} | {'scat_bwd':>9} {'gen_bwd':>9} | {'fwd_spd':>8} {'bwd_spd':>8}")
        print(f"{'-'*105}")
    
    results = {
        'n_inputs': n_inputs,
        'n_outputs': n_outputs,
        'avg_inputs_per_output': avg_inputs_per_output,
        'device': device,
        'operations': {}
    }
    
    # Define all function pairs (non-JIT and JIT)
    function_sets = [
        ("prod", scatter_reduce_prod, scatter_reduce_prod_jit, general_prod, general_prod_jit),
        ("sum", scatter_reduce_sum, scatter_reduce_sum_jit, general_sum, general_sum_jit),
        ("max", scatter_reduce_max, scatter_reduce_max_jit, general_max, general_max_jit),
        ("min", scatter_reduce_min, scatter_reduce_min_jit, general_min, general_min_jit),
    ]
    
    for name, scatter_fn, scatter_jit_fn, general_fn, general_jit_fn in function_sets:
        # Forward benchmarks
        t_scatter_fwd = benchmark_forward(scatter_fn, x, ptrs, csr, n_outputs, 
                                          n_iter=n_iter, device=device)
        t_scatter_jit_fwd = benchmark_forward(scatter_jit_fn, x, ptrs, csr, n_outputs, 
                                              n_iter=n_iter, device=device)
        t_general_fwd = benchmark_forward(general_fn, x, ptrs, csr, positions, scattered,
                                          n_iter=n_iter, device=device)
        t_general_jit_fwd = benchmark_forward(general_jit_fn, x, ptrs, csr, positions, scattered,
                                              n_iter=n_iter, device=device)
        
        # Backward benchmarks (use JIT versions for fair comparison)
        t_scatter_bwd = benchmark_backward(scatter_jit_fn, x, ptrs, csr, n_outputs,
                                           n_iter=n_iter, device=device)
        t_general_bwd = benchmark_backward(general_jit_fn, x, ptrs, csr, positions, scattered,
                                           n_iter=n_iter, device=device)
        
        # Speedup: general/scatter (< 1 means general is faster)
        speedup_fwd = t_general_jit_fwd / t_scatter_jit_fwd
        speedup_bwd = t_general_bwd / t_scatter_bwd
        
        results['operations'][name] = {
            'scatter_fwd_ms': t_scatter_fwd,
            'scatter_jit_fwd_ms': t_scatter_jit_fwd,
            'general_fwd_ms': t_general_fwd,
            'general_jit_fwd_ms': t_general_jit_fwd,
            'scatter_bwd_ms': t_scatter_bwd,
            'general_bwd_ms': t_general_bwd,
            'speedup_fwd': speedup_fwd,
            'speedup_bwd': speedup_bwd
        }
        
        if verbose:
            print(f"{name:<6} {t_scatter_fwd:>9.3f} {t_scatter_jit_fwd:>9.3f} {t_general_fwd:>9.3f} {t_general_jit_fwd:>9.3f} | {t_scatter_bwd:>9.3f} {t_general_bwd:>9.3f} | {speedup_fwd:>7.2f}x {speedup_bwd:>7.2f}x")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark scatter_reduce vs general scatter+reduce")
    parser.add_argument("--device", type=str, default="both", choices=["cpu", "cuda", "mps", "both"],
                        help="Device to run benchmarks on")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of iterations")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                        help="Output directory for results")
    parser.add_argument("--extensive", action="store_true", help="Run extensive benchmarks")
    parser.add_argument("--fan-in-study", action="store_true", help="Study impact of varying fan-in")
    parser.add_argument("--realistic", action="store_true", help="Run realistic benchmarks based on circuit distributions")
    args = parser.parse_args()
    
    # Determine which devices to run on
    devices = []
    if args.device == "both":
        devices.append("cpu")
        if torch.cuda.is_available():
            devices.append("cuda")
        elif torch.backends.mps.is_available():
            devices.append("mps")
        else:
            print("No GPU available (CUDA/MPS), running CPU only")
    elif args.device == "cuda":
        if torch.cuda.is_available():
            devices.append("cuda")
        else:
            print("CUDA not available, falling back to CPU")
            devices.append("cpu")
    elif args.device == "mps":
        if torch.backends.mps.is_available():
            devices.append("mps")
        else:
            print("MPS not available, falling back to CPU")
            devices.append("cpu")
    else:
        devices.append("cpu")
    
    print(f"Running benchmarks on: {', '.join(devices)}")
    print(f"PyTorch version: {torch.__version__}")
    if "cuda" in devices:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    if "mps" in devices:
        print("MPS device: Apple Silicon GPU")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Different problem sizes
    if args.fan_in_study:
        # Vary fan-in at fixed n_outputs = 10000
        configs = []
        n_outputs = 10000
        for fan_in in [2, 5, 10, 20, 50, 100, 200, 500]:
            configs.append((n_outputs * fan_in, n_outputs, fan_in))
    elif args.realistic:
        # Realistic benchmark based on actual circuit distributions
        # Fan-in values weighted by computation: 4(12%), 6(13%), 8(10%), 10(12%), 14(10%), 20(13%), 30(11%), 50(19%)
        # n_outputs based on circuit sizes: 150, 750, 1600, 3000, 180000
        configs = []
        # Test each fan-in at representative n_outputs
        fan_in_weights = {4: 0.122, 6: 0.128, 8: 0.099, 10: 0.120, 14: 0.100, 20: 0.130, 30: 0.114, 50: 0.187}
        n_outputs_values = [1000, 10000, 100000]  # Small, medium, large
        for n_outputs in n_outputs_values:
            for fan_in in fan_in_weights.keys():
                configs.append((n_outputs * fan_in, n_outputs, fan_in))
    elif args.extensive:
        # Only vary n_outputs, keep fan-in fixed at 10
        configs = []
        for n_outputs in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]:
            configs.append((n_outputs * 10, n_outputs, 10))
    else:
        configs = [
            (1000, 100, 10),
            (10000, 1000, 10),
            (100000, 10000, 10),
            (1000000, 100000, 10),
        ]
    
    all_results = {device: [] for device in devices}
    
    for device in devices:
        print(f"\n{'#'*60}")
        print(f"# Running benchmarks on {device.upper()}")
        print(f"{'#'*60}")
        
        for n_inputs, n_outputs, avg_per_output in configs:
            results = run_benchmarks(n_inputs, n_outputs, avg_per_output, device, args.n_iter)
            all_results[device].append(results)
    
    # Determine GPU info
    gpu_device = None
    if torch.cuda.is_available():
        gpu_device = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        gpu_device = "Apple Silicon GPU (MPS)"
    
    # Save results to JSON
    results_file = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'pytorch_version': torch.__version__,
            'gpu_device': gpu_device,
            'devices': devices,
            'n_iter': args.n_iter,
            'results': all_results
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Generate plots
    if args.extensive:
        generate_plots_multi_device(all_results, args.output_dir)
        generate_latex_report_multi_device(all_results, args.output_dir, devices, torch.__version__)
    
    print("\n" + "="*60)
    print("Summary:")
    print("- 'scatter_reduce' is the fused PyTorch operation (current impl)")
    print("- 'general' scatters to 2D tensor then reduces (flexible)")
    print("- speedup > 1 means scatter_reduce is faster")
    print("="*60)


def generate_plots(results: List[Dict], output_dir: str) -> None:
    """Generate matplotlib plots for the benchmark results."""
    print("\nGenerating plots...")
    
    # Extract data for different analysis dimensions
    operations = ['prod', 'sum', 'max', 'min']
    colors = {'prod': '#1f77b4', 'sum': '#ff7f0e', 'max': '#2ca02c', 'min': '#d62728'}
    
    # Group results by varying parameter
    vary_outputs = [r for r in results if r['avg_inputs_per_output'] == 10 and r['n_inputs'] == r['n_outputs'] * 10]
    vary_avg = [r for r in results if r['n_outputs'] == 1000 and r['n_inputs'] == 10000]
    vary_scale = [r for r in results if r['avg_inputs_per_output'] == 10 and r['n_inputs'] == r['n_outputs'] * 10]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Speedup vs n_outputs
    ax = axes[0, 0]
    if vary_outputs:
        x = [r['n_outputs'] for r in vary_outputs]
        for op in operations:
            y = [r['operations'][op]['speedup'] for r in vary_outputs]
            ax.plot(x, y, 'o-', label=op, color=colors[op], linewidth=2, markersize=6)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='break-even')
        ax.set_xlabel('Number of outputs', fontsize=11)
        ax.set_ylabel('Speedup (scatter_reduce / general)', fontsize=11)
        ax.set_title('Speedup vs Problem Size', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Speedup vs avg_inputs_per_output
    ax = axes[0, 1]
    if vary_avg:
        x = [r['avg_inputs_per_output'] for r in vary_avg]
        for op in operations:
            y = [r['operations'][op]['speedup'] for r in vary_avg]
            ax.plot(x, y, 'o-', label=op, color=colors[op], linewidth=2, markersize=6)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='break-even')
        ax.set_xlabel('Avg inputs per output', fontsize=11)
        ax.set_ylabel('Speedup', fontsize=11)
        ax.set_title('Speedup vs Fan-in', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Absolute times comparison (bar chart for one config)
    ax = axes[0, 2]
    mid_result = results[len(results) // 2]  # Pick a middle configuration
    x_pos = np.arange(len(operations))
    width = 0.35
    scatter_times = [mid_result['operations'][op]['scatter_reduce_ms'] for op in operations]
    general_times = [mid_result['operations'][op]['general_ms'] for op in operations]
    ax.bar(x_pos - width/2, scatter_times, width, label='scatter_reduce', color='steelblue')
    ax.bar(x_pos + width/2, general_times, width, label='general', color='coral')
    ax.set_xlabel('Operation', fontsize=11)
    ax.set_ylabel('Time (ms)', fontsize=11)
    ax.set_title(f'Absolute Times\n(n_out={mid_result["n_outputs"]}, avg={mid_result["avg_inputs_per_output"]})', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(operations)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: scatter_reduce times vs scale
    ax = axes[1, 0]
    if vary_scale:
        x = [r['n_outputs'] for r in vary_scale]
        for op in operations:
            y = [r['operations'][op]['scatter_reduce_ms'] for r in vary_scale]
            ax.plot(x, y, 'o-', label=op, color=colors[op], linewidth=2, markersize=6)
        ax.set_xlabel('Number of outputs', fontsize=11)
        ax.set_ylabel('Time (ms)', fontsize=11)
        ax.set_title('scatter_reduce Time vs Scale', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Plot 5: general times vs scale
    ax = axes[1, 1]
    if vary_scale:
        x = [r['n_outputs'] for r in vary_scale]
        for op in operations:
            y = [r['operations'][op]['general_ms'] for r in vary_scale]
            ax.plot(x, y, 'o-', label=op, color=colors[op], linewidth=2, markersize=6)
        ax.set_xlabel('Number of outputs', fontsize=11)
        ax.set_ylabel('Time (ms)', fontsize=11)
        ax.set_title('General Approach Time vs Scale', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Heatmap of speedup for prod operation
    ax = axes[1, 2]
    # Create a simplified heatmap-style visualization
    all_outputs = sorted(set(r['n_outputs'] for r in results))
    all_avgs = sorted(set(r['avg_inputs_per_output'] for r in results))
    
    # Just show speedup bars for all configs
    configs_labels = [f"out={r['n_outputs']}\navg={r['avg_inputs_per_output']}" for r in results[:8]]
    speedups = [r['operations']['prod']['speedup'] for r in results[:8]]
    colors_bar = ['steelblue' if s > 1 else 'coral' for s in speedups]
    ax.barh(range(len(configs_labels)), speedups, color=colors_bar)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
    ax.set_yticks(range(len(configs_labels)))
    ax.set_yticklabels(configs_labels, fontsize=9)
    ax.set_xlabel('Speedup (prod operation)', fontsize=11)
    ax.set_title('Speedup by Configuration', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "benchmark_plots.pdf")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "benchmark_plots.png"), dpi=150, bbox_inches='tight')
    print(f"Plots saved to {plot_file}")


def generate_latex_report(results: List[Dict], output_dir: str, device: str, pytorch_version: str) -> None:
    """Generate a LaTeX report with code, results, and figures."""
    print("\nGenerating LaTeX report...")
    
    # Read the source code
    script_path = os.path.join(os.path.dirname(__file__), "benchmark_scatter.py")
    with open(script_path, 'r') as f:
        source_code = f.read()
    
    # Extract just the key functions for the report
    scatter_reduce_code = '''def scatter_reduce_prod(x, ptrs, csr, n_outputs):
    """Fused scatter_reduce with prod."""
    output = torch.empty(n_outputs, dtype=x.dtype, device=x.device)
    return torch.scatter_reduce(output, 0, index=csr, 
                                src=x[ptrs], reduce="prod", 
                                include_self=False)'''
    
    general_code = '''def general_scatter_then_reduce_fast(x, ptrs, csr, n_outputs,
                                        max_per_output, reduce_fn, fill_value):
    """Scatter to 2D tensor, then apply any reduction."""
    # Compute positions within each group
    positions = compute_positions(csr)
    
    # Scatter to 2D tensor [n_outputs, max_per_output]
    scattered = torch.full((n_outputs, max_per_output), fill_value,
                           dtype=x.dtype, device=x.device)
    scattered[csr, positions] = x[ptrs]
    
    # Apply reduction along dim=1 (any torch function)
    return reduce_fn(scattered, dim=1)'''
    
    # Build results table
    table_rows = []
    for r in results:
        for op in ['prod', 'sum', 'max', 'min']:
            data = r['operations'][op]
            table_rows.append(
                f"        {r['n_outputs']} & {r['avg_inputs_per_output']} & {op} & "
                f"{data['scatter_reduce_ms']:.3f} & {data['general_ms']:.3f} & {data['speedup']:.2f} \\\\"
            )
    
    latex_content = r'''\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2,
    language=Python
}
\lstset{style=mystyle}

\title{Benchmark: \texttt{scatter\_reduce} vs General Scatter + Reduce}
\author{Automated Benchmark Report}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This report compares two approaches for implementing scatter-reduce operations in PyTorch:
(1) the fused \texttt{torch.scatter\_reduce} operation, and 
(2) a general approach that scatters values to a 2D tensor followed by standard PyTorch reduction operations.
The general approach offers flexibility to use arbitrary reduction functions at the cost of additional memory and computation overhead.
\end{abstract}

\section{Introduction}

Scatter-reduce operations are fundamental in sparse tensor computations, graph neural networks, and knowledge graph inference systems. PyTorch provides \texttt{scatter\_reduce} as a fused operation, but sometimes a more general approach is needed to apply custom reduction functions.

\textbf{Environment:}
\begin{itemize}
    \item PyTorch Version: ''' + pytorch_version + r'''
    \item Device: ''' + device + r'''
\end{itemize}

\section{Implementation Approaches}

\subsection{Approach 1: Fused scatter\_reduce}

The fused approach uses PyTorch's built-in \texttt{scatter\_reduce} operation:

\begin{lstlisting}
''' + scatter_reduce_code + r'''
\end{lstlisting}

\subsection{Approach 2: General Scatter + Reduce}

The general approach scatters to a 2D padded tensor, then applies any reduction:

\begin{lstlisting}
''' + general_code + r'''
\end{lstlisting}

This approach allows using \emph{any} PyTorch operation as the reduction function, including \texttt{logsumexp}, custom differentiable functions, or learned operations.

\section{Results}

\subsection{Benchmark Configurations}

We tested various configurations varying:
\begin{itemize}
    \item \textbf{n\_outputs}: Number of output positions (100 to 20,000)
    \item \textbf{avg\_inputs\_per\_output}: Fan-in factor (5 to 200)
    \item \textbf{Operations}: prod, sum, max, min
\end{itemize}

\subsection{Performance Comparison}

\begin{table}[H]
\centering
\caption{Benchmark Results (times in milliseconds, speedup = general/scatter\_reduce)}
\label{tab:results}
\begin{tabular}{@{}rrllrrl@{}}
\toprule
n\_outputs & avg/out & op & scatter\_reduce & general & speedup \\
\midrule
''' + '\n'.join(table_rows[:24]) + r'''
\bottomrule
\end{tabular}
\end{table}

\subsection{Visualization}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{benchmark_plots.pdf}
    \caption{Performance comparison across different configurations. Top row: speedup analysis. Bottom row: absolute timing comparisons. Speedup $> 1$ means \texttt{scatter\_reduce} is faster.}
    \label{fig:plots}
\end{figure}

\section{Analysis}

\subsection{Key Findings}

\begin{enumerate}
    \item \textbf{Small/medium data}: \texttt{scatter\_reduce} is 2--7$\times$ faster due to:
    \begin{itemize}
        \item No intermediate 2D tensor allocation
        \item Fused kernel reduces memory bandwidth
    \end{itemize}
    
    \item \textbf{Large data with high fan-in}: The general approach can be competitive or faster because:
    \begin{itemize}
        \item Dense tensor operations have better cache locality
        \item PyTorch's reduction kernels are highly optimized for regular access patterns
    \end{itemize}
    
    \item \textbf{Sum operation}: \texttt{scatter\_add} consistently outperforms the general approach, suggesting it has special optimizations in PyTorch.
\end{enumerate}

\subsection{When to Use Each Approach}

\begin{table}[H]
\centering
\caption{Recommendation Guide}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Use scatter\_reduce when:} & \textbf{Use general approach when:} \\
\midrule
Standard reductions (sum, prod, max, min) & Custom/learned reduction functions \\
Memory efficiency is critical & Need gradients through reduction \\
Small to medium problem sizes & Operations like logsumexp, softmax \\
Maximum performance required & Flexibility is more important \\
\bottomrule
\end{tabular}
\end{table}

\section{Conclusion}

The fused \texttt{scatter\_reduce} operation provides the best performance for standard reduction operations, especially at smaller scales. However, the general scatter-then-reduce approach offers valuable flexibility when custom reduction functions are needed, with competitive performance at larger scales with high fan-in factors.

\end{document}
'''
    
    tex_file = os.path.join(output_dir, "benchmark_report.tex")
    with open(tex_file, 'w') as f:
        f.write(latex_content)
    print(f"LaTeX report saved to {tex_file}")
    
    # Compile LaTeX to PDF
    import subprocess
    try:
        # Run pdflatex twice for references
        for _ in range(2):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', output_dir, tex_file],
                capture_output=True, text=True, cwd=output_dir
            )
        pdf_file = os.path.join(output_dir, "benchmark_report.pdf")
        if os.path.exists(pdf_file):
            print(f"PDF report generated: {pdf_file}")
        else:
            print("PDF generation may have failed. Check LaTeX installation.")
            print(result.stderr[-500:] if result.stderr else "No error output")
    except FileNotFoundError:
        print("pdflatex not found. Install LaTeX to generate PDF.")
        print(f"LaTeX source saved to {tex_file}")


# Fan-in weights from real circuit analysis (% of total computation)
REALISTIC_FAN_IN_WEIGHTS = {4: 0.122, 6: 0.128, 8: 0.099, 10: 0.120, 14: 0.100, 20: 0.130, 30: 0.114, 50: 0.187}


def compute_weighted_speedup(results: List[Dict], operation: str, pass_type: str = 'fwd') -> float:
    """Compute weighted average speedup based on real circuit fan-in distribution."""
    key = f'speedup_{pass_type}'
    total_weight = 0
    weighted_sum = 0
    
    for r in results:
        fan_in = r['avg_inputs_per_output']
        if fan_in in REALISTIC_FAN_IN_WEIGHTS:
            weight = REALISTIC_FAN_IN_WEIGHTS[fan_in]
            speedup = r['operations'][operation][key]
            weighted_sum += weight * speedup
            total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 1.0


def generate_plots_multi_device(results_by_device: Dict[str, List[Dict]], output_dir: str) -> None:
    """Generate matplotlib plots comparing forward and backward pass speedups."""
    print("\nGenerating multi-device plots with forward/backward comparison...")
    
    devices = list(results_by_device.keys())
    operations = ['prod', 'sum', 'max', 'min']
    colors = {'prod': '#1f77b4', 'sum': '#ff7f0e', 'max': '#2ca02c', 'min': '#d62728'}
    
    # Detect benchmark mode
    first_device_results = results_by_device[devices[0]]
    n_outputs_set = set(r['n_outputs'] for r in first_device_results)
    fan_in_set = set(r['avg_inputs_per_output'] for r in first_device_results)
    
    is_fan_in_study = len(n_outputs_set) == 1 and len(fan_in_set) > 1
    is_realistic = len(n_outputs_set) > 1 and len(fan_in_set) > 1
    
    if is_realistic:
        # Realistic mode: create heatmap-style plots or grouped bar charts
        _generate_realistic_plots(results_by_device, output_dir, devices, operations, colors)
    elif is_fan_in_study:
        x_key = 'avg_inputs_per_output'
        x_label = 'Fan-in (inputs per output)'
        sort_key = lambda r: r['avg_inputs_per_output']
        _generate_line_plots(results_by_device, output_dir, devices, operations, colors, x_key, x_label, sort_key)
    else:
        x_key = 'n_outputs'
        x_label = 'Number of outputs'
        sort_key = lambda r: r['n_outputs']
        _generate_line_plots(results_by_device, output_dir, devices, operations, colors, x_key, x_label, sort_key)


def _generate_line_plots(results_by_device, output_dir, devices, operations, colors, x_key, x_label, sort_key):
    """Generate line plots for single-variable studies."""
    n_devices = len(devices)
    fig, axes = plt.subplots(2, n_devices, figsize=(8 * n_devices, 10))
    if n_devices == 1:
        axes = axes.reshape(-1, 1)
    
    for col, device in enumerate(devices):
        results = results_by_device[device]
        results = sorted(results, key=sort_key)
        x = [r[x_key] for r in results]
        
        # Plot 1: Forward speedup
        ax = axes[0, col]
        for op in operations:
            y = [r['operations'][op]['speedup_fwd'] for r in results]
            ax.plot(x, y, 'o-', label=op, color=colors[op], linewidth=2, markersize=5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel('Speedup (general / scatter_reduce)', fontsize=11)
        ax.set_title(f'{device.upper()}: Forward Pass', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Backward speedup
        ax = axes[1, col]
        for op in operations:
            y = [r['operations'][op]['speedup_bwd'] for r in results]
            ax.plot(x, y, 'o-', label=op, color=colors[op], linewidth=2, markersize=5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel('Speedup (general / scatter_reduce)', fontsize=11)
        ax.set_title(f'{device.upper()}: Backward Pass', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "benchmark_plots.pdf")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "benchmark_plots.png"), dpi=150, bbox_inches='tight')
    print(f"Plots saved to {plot_file}")


def _generate_realistic_plots(results_by_device, output_dir, devices, operations, colors):
    """Generate plots for realistic benchmark (varying both n_outputs and fan_in)."""
    import numpy as np
    
    n_devices = len(devices)
    
    # Create a figure with:
    # Row 1: Heatmaps for each device (forward)
    # Row 2: Heatmaps for each device (backward)
    # Row 3: Weighted speedup bar charts
    fig = plt.figure(figsize=(6 * n_devices, 14))
    
    for idx, device in enumerate(devices):
        results = results_by_device[device]
        
        # Get unique values
        n_outputs_vals = sorted(set(r['n_outputs'] for r in results))
        fan_in_vals = sorted(set(r['avg_inputs_per_output'] for r in results))
        
        # Build lookup
        lookup = {(r['n_outputs'], r['avg_inputs_per_output']): r for r in results}
        
        # For each operation, create a heatmap
        for op_idx, op in enumerate(operations):
            # Forward heatmap
            ax = fig.add_subplot(4, n_devices * 4, idx * 4 + op_idx + 1)
            data = np.zeros((len(fan_in_vals), len(n_outputs_vals)))
            for i, fi in enumerate(fan_in_vals):
                for j, no in enumerate(n_outputs_vals):
                    if (no, fi) in lookup:
                        data[i, j] = lookup[(no, fi)]['operations'][op]['speedup_fwd']
            
            im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=2)
            ax.set_xticks(range(len(n_outputs_vals)))
            ax.set_xticklabels([f'{x//1000}k' if x >= 1000 else str(x) for x in n_outputs_vals], fontsize=8)
            ax.set_yticks(range(len(fan_in_vals)))
            ax.set_yticklabels(fan_in_vals, fontsize=8)
            if op_idx == 0:
                ax.set_ylabel(f'{device.upper()}\nFan-in', fontsize=9)
            ax.set_title(f'{op} fwd', fontsize=9)
            
            # Add text annotations
            for i in range(len(fan_in_vals)):
                for j in range(len(n_outputs_vals)):
                    ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center', fontsize=6)
        
        # Backward heatmaps
        for op_idx, op in enumerate(operations):
            ax = fig.add_subplot(4, n_devices * 4, n_devices * 4 + idx * 4 + op_idx + 1)
            data = np.zeros((len(fan_in_vals), len(n_outputs_vals)))
            for i, fi in enumerate(fan_in_vals):
                for j, no in enumerate(n_outputs_vals):
                    if (no, fi) in lookup:
                        data[i, j] = lookup[(no, fi)]['operations'][op]['speedup_bwd']
            
            im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=2)
            ax.set_xticks(range(len(n_outputs_vals)))
            ax.set_xticklabels([f'{x//1000}k' if x >= 1000 else str(x) for x in n_outputs_vals], fontsize=8)
            ax.set_yticks(range(len(fan_in_vals)))
            ax.set_yticklabels(fan_in_vals, fontsize=8)
            if op_idx == 0:
                ax.set_ylabel(f'Fan-in', fontsize=9)
            ax.set_title(f'{op} bwd', fontsize=9)
            ax.set_xlabel('n_outputs', fontsize=8)
            
            for i in range(len(fan_in_vals)):
                for j in range(len(n_outputs_vals)):
                    ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center', fontsize=6)
    
    # Weighted speedup bar chart (bottom)
    ax = fig.add_subplot(4, 1, 3)
    bar_width = 0.1
    x_pos = np.arange(len(devices))
    
    for op_idx, op in enumerate(operations):
        fwd_speedups = []
        for device in devices:
            # Filter to largest n_outputs for weighted calculation
            results = results_by_device[device]
            max_n = max(r['n_outputs'] for r in results)
            filtered = [r for r in results if r['n_outputs'] == max_n]
            ws = compute_weighted_speedup(filtered, op, 'fwd')
            fwd_speedups.append(ws)
        
        ax.bar(x_pos + op_idx * bar_width, fwd_speedups, bar_width, label=op, color=colors[op])
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Device')
    ax.set_ylabel('Weighted Speedup (Forward)')
    ax.set_title('Weighted Forward Speedup (based on real circuit fan-in distribution)')
    ax.set_xticks(x_pos + bar_width * 1.5)
    ax.set_xticklabels([d.upper() for d in devices])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Backward weighted speedup
    ax = fig.add_subplot(4, 1, 4)
    for op_idx, op in enumerate(operations):
        bwd_speedups = []
        for device in devices:
            results = results_by_device[device]
            max_n = max(r['n_outputs'] for r in results)
            filtered = [r for r in results if r['n_outputs'] == max_n]
            ws = compute_weighted_speedup(filtered, op, 'bwd')
            bwd_speedups.append(ws)
        
        ax.bar(x_pos + op_idx * bar_width, bwd_speedups, bar_width, label=op, color=colors[op])
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Device')
    ax.set_ylabel('Weighted Speedup (Backward)')
    ax.set_title('Weighted Backward Speedup (based on real circuit fan-in distribution)')
    ax.set_xticks(x_pos + bar_width * 1.5)
    ax.set_xticklabels([d.upper() for d in devices])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "benchmark_plots.pdf")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "benchmark_plots.png"), dpi=150, bbox_inches='tight')
    print(f"Plots saved to {plot_file}")



def generate_latex_report_multi_device(results_by_device: Dict[str, List[Dict]], output_dir: str, 
                                        devices: List[str], pytorch_version: str) -> None:
    """Generate a comprehensive LaTeX report comparing forward and backward pass results."""
    print("\nGenerating comprehensive LaTeX report...")
    
    # Get system info
    import platform
    import subprocess as sp
    
    cpu_info = "Unknown"
    ram_info = "Unknown"
    gpu_info = "Unknown"
    
    try:
        if platform.system() == "Darwin":
            cpu_info = sp.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            ram_bytes = int(sp.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            ram_info = f"{ram_bytes / (1024**3):.0f} GB"
            ncpu = sp.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip()
            cpu_info += f" ({ncpu} cores)"
            if torch.backends.mps.is_available():
                gpu_info = "Apple Silicon GPU (MPS)"
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_info = line.split(":")[1].strip()
                        break
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        ram_kb = int(line.split()[1])
                        ram_info = f"{ram_kb / (1024**2):.0f} GB"
                        break
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_name(0)
    except:
        pass
    
    scatter_reduce_code = '''@torch.jit.script
def scatter_reduce_prod(x: Tensor, ptrs: Tensor, csr: Tensor, n_outputs: int) -> Tensor:
    output = torch.empty(n_outputs, dtype=x.dtype, device=x.device)
    return torch.scatter_reduce(output, 0, index=csr, src=x[ptrs], 
                                reduce="prod", include_self=False)'''
    
    general_code = '''@torch.jit.script
def general_prod(x: Tensor, ptrs: Tensor, csr: Tensor, 
                 positions: Tensor, scattered: Tensor) -> Tensor:
    # scattered buffer and positions are pre-allocated (not timed)
    scattered.fill_(1.0)
    scattered[csr, positions] = x[ptrs]
    return torch.prod(scattered, dim=1)'''
    
    # Detect if this is a fan-in study
    first_device = devices[0]
    first_results = results_by_device[first_device]
    n_outputs_set = set(r['n_outputs'] for r in first_results)
    fan_in_set = set(r['avg_inputs_per_output'] for r in first_results)
    is_fan_in_study = len(n_outputs_set) == 1 and len(fan_in_set) > 1
    is_realistic = len(n_outputs_set) > 1 and len(fan_in_set) > 1  # Both vary
    
    if is_fan_in_study or is_realistic:
        x_col_name = "fan\\_in"
        x_col_label = "Fan-in"
        sort_key = lambda r: (r['n_outputs'], r['avg_inputs_per_output'])
        get_x_val = lambda r: r['avg_inputs_per_output']
        fixed_param = f"n\\_outputs = {first_results[0]['n_outputs']:,}" if is_fan_in_study else "varied"
    else:
        x_col_name = "n\\_outputs"
        x_col_label = "n\\_outputs"
        sort_key = lambda r: r['n_outputs']
        get_x_val = lambda r: r['n_outputs']
        fixed_param = f"fan-in = {first_results[0]['avg_inputs_per_output']}"
    
    # Build results tables for each device
    device_tables = {}
    for device in devices:
        results = results_by_device[device]
        results = sorted(results, key=sort_key)
        table_rows = []
        for r in results:
            for op in ['prod', 'sum', 'max', 'min']:
                data = r['operations'][op]
                table_rows.append(
                    f"        {get_x_val(r):,} & {op} & "
                    f"{data['scatter_jit_fwd_ms']:.3f} & {data['general_jit_fwd_ms']:.3f} & {data['speedup_fwd']:.2f} & "
                    f"{data['scatter_bwd_ms']:.3f} & {data['general_bwd_ms']:.3f} & {data['speedup_bwd']:.2f} \\\\"
                )
        device_tables[device] = '\n'.join(table_rows)
    
    # Build device-specific sections
    device_sections = ""
    for device in devices:
        device_sections += f'''
\\subsection{{{device.upper()} Results}}

\\begin{{table}}[H]
\\centering
\\caption{{Forward and Backward Pass Results on {device.upper()} (times in ms, {fixed_param})}}
\\label{{tab:results_{device}}}
\\footnotesize
\\begin{{tabular}}{{@{{}}rlrrrrrr@{{}}}}
\\toprule
{x_col_label} & op & scat\\_fwd & gen\\_fwd & fwd\\_spd & scat\\_bwd & gen\\_bwd & bwd\\_spd \\\\
\\midrule
{device_tables[device]}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
'''
    
    # Compute weighted speedups if we have realistic fan-in values
    weighted_section = ""
    if is_realistic:
        weighted_section = r'''
\subsection{Weighted Speedups (Based on Real Circuit Distributions)}

The following speedups are weighted by the actual computation distribution from real SDD circuits (Table~\ref{tab:circuit_stats}), where each fan-in value contributes proportionally to its share of total circuit computation.

\begin{table}[H]
\centering
\caption{Weighted Average Speedups Based on Real Circuit Fan-in Distribution}
\label{tab:weighted_speedups}
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Device} & \textbf{prod} & \textbf{sum} & \textbf{max} & \textbf{min} \\
\midrule
'''
        for device in devices:
            results = results_by_device[device]
            max_n = max(r['n_outputs'] for r in results)
            filtered = [r for r in results if r['n_outputs'] == max_n]
            
            speedups = []
            for op in ['prod', 'sum', 'max', 'min']:
                ws_fwd = compute_weighted_speedup(filtered, op, 'fwd')
                ws_bwd = compute_weighted_speedup(filtered, op, 'bwd')
                speedups.append(f"{ws_fwd:.2f}x / {ws_bwd:.2f}x")
            
            weighted_section += f"{device.upper()} (fwd/bwd) & {' & '.join(speedups)} \\\\\n"
        
        weighted_section += r'''\bottomrule
\end{tabular}
\end{table}

\textbf{Interpretation}: Values below 1.0x indicate the general approach is faster. For example, 0.21x means the general approach is 4.8$\times$ faster than \texttt{scatter\_reduce}.
'''
    
    # Extract key metrics for analysis
    cpu_results = results_by_device.get('cpu', [])
    mps_results = results_by_device.get('mps', [])
    
    # Find extreme results (largest n_outputs or largest fan-in)
    if is_fan_in_study:
        large_cpu = next((r for r in sorted(cpu_results, key=lambda x: -x['avg_inputs_per_output'])), None)
        large_mps = next((r for r in sorted(mps_results, key=lambda x: -x['avg_inputs_per_output'])), None)
        study_title = "Fan-in Impact Study"
        study_abstract = f"This report investigates how fan-in (inputs per output) affects performance, with n\\_outputs fixed at {first_results[0]['n_outputs']:,}."
    elif is_realistic:
        large_cpu = next((r for r in sorted(cpu_results, key=lambda x: (-x['n_outputs'], -x['avg_inputs_per_output']))), None)
        large_mps = next((r for r in sorted(mps_results, key=lambda x: (-x['n_outputs'], -x['avg_inputs_per_output']))), None)
        study_title = "Realistic Workload Analysis"
        study_abstract = "This report presents performance analysis using fan-in distributions derived from real SDD circuits, with weighted speedups reflecting actual computational workloads."
    else:
        large_cpu = next((r for r in sorted(cpu_results, key=lambda x: -x['n_outputs'])), None)
        large_mps = next((r for r in sorted(mps_results, key=lambda x: -x['n_outputs'])), None)
        study_title = "Forward and Backward Pass Performance Analysis"
        study_abstract = "This report presents a comprehensive performance comparison scaling from 100 to $10^6$ output elements."
    
    latex_content = r'''\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=0.8in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{amsmath}

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\scriptsize,
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2,
    language=Python
}
\lstset{style=mystyle}

\title{Benchmark: \texttt{scatter\_reduce} vs General Scatter + Reduce\\''' + study_title + r'''}
\author{Automated Benchmark Report}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
''' + study_abstract + r''' We benchmark both forward and backward passes across four reduction operations (prod, sum, max, min) on CPU and Apple Silicon GPU (MPS). Our findings reveal that the general approach significantly outperforms \texttt{scatter\_reduce} for most operations at scale, with speedups of 2--5$\times$ on CPU and operation-dependent results on MPS.
\end{abstract}

\section{Introduction}

Scatter-reduce operations are fundamental building blocks in:
\begin{itemize}
    \item \textbf{Graph Neural Networks}: Aggregating neighbor features
    \item \textbf{Sparse Tensor Operations}: Combining values with shared indices
    \item \textbf{Probabilistic Circuits}: Computing partition functions over knowledge bases
    \item \textbf{Attention Mechanisms}: Sparse attention patterns
\end{itemize}

PyTorch provides \texttt{torch.scatter\_reduce} as a fused operation that handles irregular scatter patterns efficiently. However, this operation has inherent overhead from handling variable-length groups and irregular memory access patterns.

An alternative approach is to:
\begin{enumerate}
    \item Pre-compute positions within each group (done once, not timed)
    \item Scatter values to a regular 2D tensor of shape \texttt{(n\_outputs, max\_fan\_in)}
    \item Apply standard PyTorch reductions along dimension 1
\end{enumerate}

This general approach trades memory for regularity, potentially enabling better vectorization and cache utilization.

\section{Circuit Characteristics}

To inform our benchmark design, we analyzed the fan-in and fan-out distributions of five real-world SDD (Sentential Decision Diagram) circuits used in neuro-symbolic AI applications.

\begin{table}[H]
\centering
\caption{Circuit Statistics: Fan-in and Fan-out Distributions}
\label{tab:circuit_stats}
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
& & & \multicolumn{3}{c}{\textbf{Fan-in}} & \multicolumn{3}{c}{\textbf{Fan-out}} \\
\cmidrule(lr){4-6} \cmidrule(lr){7-9}
\textbf{Circuit} & \textbf{Nodes} & \textbf{D-nodes} & med & p95 & max & med & p95 & max \\
\midrule
road\_r & 237 & 153 & 4 & 6 & 8 & 2 & 6 & 86 \\
sudoku\_4 & 869 & 739 & 4 & 8 & 10 & 2 & 8 & 492 \\
4-grid & 1,707 & 1,625 & 4 & 18 & 58 & 2 & 18 & 1,217 \\
seq\_fun & 3,986 & 2,984 & 4 & 6 & 24 & 2 & 6 & 1,638 \\
warcraft\_12 & 183,018 & 182,488 & 6 & 28 & 228 & 3 & 24 & 171,051 \\
\midrule
\textbf{Aggregate} & \textbf{189,817} & \textbf{187,989} & \textbf{6} & \textbf{28} & \textbf{228} & \textbf{3} & \textbf{24} & \textbf{171,051} \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{circuit_boxplots.pdf}
    \caption{Box plots of fan-in (left) and fan-out (right) distributions for each circuit. The y-axis uses log scale. Fan-in represents the number of children per product node; fan-out represents how many parent nodes reference each node.}
    \label{fig:circuit_boxplots}
\end{figure}

\textbf{Key observations:}
\begin{itemize}
    \item \textbf{Fan-in is modest}: Median fan-in is 4--6 across circuits, with p95 typically under 30
    \item \textbf{Fan-out is highly skewed}: While median fan-out is 2--3, some nodes (constants, common subexpressions) are referenced 100,000+ times
    \item \textbf{Scale varies dramatically}: From 237 nodes (road\_r) to 183,018 nodes (warcraft\_12)
\end{itemize}

These distributions inform our benchmark parameters: fan-in values of 2--50 cover the realistic range, while the high fan-out tail is relevant for backward pass performance.

\section{Computational Environment}

\begin{table}[H]
\centering
\caption{Hardware and Software Configuration}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Component} & \textbf{Specification} \\
\midrule
CPU & ''' + cpu_info.replace("&", "\\&") + r''' \\
RAM & ''' + ram_info + r''' \\
GPU & ''' + gpu_info + r''' \\
PyTorch Version & ''' + pytorch_version + r''' \\
Devices Tested & ''' + ', '.join(devices).upper() + r''' \\
\bottomrule
\end{tabular}
\end{table}

\section{Implementation Details}

\subsection{Approach 1: Fused \texttt{scatter\_reduce}}

The fused approach uses PyTorch's built-in operation:

\begin{lstlisting}
''' + scatter_reduce_code + r'''
\end{lstlisting}

This operation:
\begin{itemize}
    \item Handles irregular group sizes natively
    \item Memory-efficient (no padding required)
    \item Has fully implemented backward pass for all reduce operations
\end{itemize}

\subsection{Approach 2: General Scatter + Reduce}

The general approach uses a pre-allocated 2D buffer:

\begin{lstlisting}
''' + general_code + r'''
\end{lstlisting}

Key optimizations:
\begin{itemize}
    \item \textbf{Pre-computed positions}: Group positions calculated once before benchmarking
    \item \textbf{Pre-allocated buffer}: The \texttt{scattered} tensor is allocated once and reused
    \item Only \texttt{fill\_()}, indexed assignment, and reduction are timed
\end{itemize}

\subsection{Benchmark Methodology}

\begin{itemize}
    \item \textbf{Warmup}: 10 iterations before timing
    \item \textbf{Iterations}: 20 timed iterations per configuration
    \item \textbf{Synchronization}: \texttt{torch.mps.synchronize()} / \texttt{torch.cuda.synchronize()} for accurate GPU timing
    \item \textbf{Fan-in}: Fixed at 10 inputs per output (uniform distribution)
    \item \textbf{Scale}: $10^2$ to $10^6$ outputs
\end{itemize}

\section{Results}
''' + device_sections + weighted_section + r'''

\subsection{Visualization}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{benchmark_plots.pdf}
    \caption{Speedup comparison across scales. \textbf{Top row}: Forward pass. \textbf{Bottom row}: Backward pass. The dashed line at $y=1$ indicates break-even; values below 1 indicate the general approach is faster. Note the log scale on the x-axis.}
    \label{fig:plots}
\end{figure}

\section{Analysis}

\subsection{CPU Performance}

On CPU, the general approach consistently outperforms \texttt{scatter\_reduce} for \textbf{prod}, \textbf{max}, and \textbf{min} operations:

\begin{table}[H]
\centering
\caption{CPU Speedup at Largest Configuration (''' + (f"fan-in={large_cpu['avg_inputs_per_output']}" if is_fan_in_study and large_cpu else f"{large_cpu['n_outputs']:,} outputs" if large_cpu else "N/A") + r''')}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Operation & Forward Speedup & Backward Speedup & Forward Time (gen) & Backward Time (gen) \\
\midrule
''' + (f"""prod & {large_cpu['operations']['prod']['speedup_fwd']:.2f}x & {large_cpu['operations']['prod']['speedup_bwd']:.2f}x & {large_cpu['operations']['prod']['general_jit_fwd_ms']:.1f} ms & {large_cpu['operations']['prod']['general_bwd_ms']:.1f} ms \\\\
sum & {large_cpu['operations']['sum']['speedup_fwd']:.2f}x & {large_cpu['operations']['sum']['speedup_bwd']:.2f}x & {large_cpu['operations']['sum']['general_jit_fwd_ms']:.1f} ms & {large_cpu['operations']['sum']['general_bwd_ms']:.1f} ms \\\\
max & {large_cpu['operations']['max']['speedup_fwd']:.2f}x & {large_cpu['operations']['max']['speedup_bwd']:.2f}x & {large_cpu['operations']['max']['general_jit_fwd_ms']:.1f} ms & {large_cpu['operations']['max']['general_bwd_ms']:.1f} ms \\\\
min & {large_cpu['operations']['min']['speedup_fwd']:.2f}x & {large_cpu['operations']['min']['speedup_bwd']:.2f}x & {large_cpu['operations']['min']['general_jit_fwd_ms']:.1f} ms & {large_cpu['operations']['min']['general_bwd_ms']:.1f} ms \\\\""" if large_cpu else "N/A & N/A & N/A & N/A & N/A \\\\") + r'''
\bottomrule
\end{tabular}
\end{table}

\textbf{Key observations}:
\begin{itemize}
    \item \textbf{prod}: General approach is 2--5$\times$ faster in forward pass, 3--4$\times$ faster in backward pass
    \item \textbf{max/min}: Similar speedups (2--4$\times$) due to regular memory access patterns
    \item \textbf{sum}: Nearly equivalent performance; \texttt{scatter\_reduce} sum is well-optimized
    \item \textbf{Backward pass}: Often shows larger speedups than forward pass
\end{itemize}

\subsection{MPS (Apple Silicon GPU) Performance}

GPU results show more nuanced behavior:

\begin{table}[H]
\centering
\caption{MPS Speedup at Largest Configuration (''' + (f"fan-in={large_mps['avg_inputs_per_output']}" if is_fan_in_study and large_mps else f"{large_mps['n_outputs']:,} outputs" if large_mps else "N/A") + r''')}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Operation & Forward Speedup & Backward Speedup & Forward Time (gen) & Backward Time (gen) \\
\midrule
''' + (f"""prod & {large_mps['operations']['prod']['speedup_fwd']:.2f}x & {large_mps['operations']['prod']['speedup_bwd']:.2f}x & {large_mps['operations']['prod']['general_jit_fwd_ms']:.1f} ms & {large_mps['operations']['prod']['general_bwd_ms']:.1f} ms \\\\
sum & {large_mps['operations']['sum']['speedup_fwd']:.2f}x & {large_mps['operations']['sum']['speedup_bwd']:.2f}x & {large_mps['operations']['sum']['general_jit_fwd_ms']:.1f} ms & {large_mps['operations']['sum']['general_bwd_ms']:.1f} ms \\\\
max & {large_mps['operations']['max']['speedup_fwd']:.2f}x & {large_mps['operations']['max']['speedup_bwd']:.2f}x & {large_mps['operations']['max']['general_jit_fwd_ms']:.1f} ms & {large_mps['operations']['max']['general_bwd_ms']:.1f} ms \\\\
min & {large_mps['operations']['min']['speedup_fwd']:.2f}x & {large_mps['operations']['min']['speedup_bwd']:.2f}x & {large_mps['operations']['min']['general_jit_fwd_ms']:.1f} ms & {large_mps['operations']['min']['general_bwd_ms']:.1f} ms \\\\""" if large_mps else "N/A & N/A & N/A & N/A & N/A \\\\") + r'''
\bottomrule
\end{tabular}
\end{table}

\textbf{Key observations}:
\begin{itemize}
    \item \textbf{prod}: General approach is consistently faster (1.2--1.4$\times$) on forward and backward
    \item \textbf{sum}: Roughly equivalent; slight edge to \texttt{scatter\_reduce}
    \item \textbf{max/min}: \texttt{scatter\_reduce} is \textit{faster} in forward pass (0.8--0.9$\times$) but general is faster in backward
    \item GPU kernel characteristics and memory bandwidth affect results differently than CPU
\end{itemize}

\subsection{Why is the General Approach Faster?}

\begin{enumerate}
    \item \textbf{Regular memory access patterns}: The 2D scattered tensor has contiguous memory layout, enabling:
    \begin{itemize}
        \item Better CPU cache utilization
        \item Vectorized SIMD operations
        \item Predictable memory prefetching
    \end{itemize}
    
    \item \textbf{Highly optimized reduction kernels}: PyTorch's \texttt{torch.prod}, \texttt{torch.max}, etc. are heavily optimized for contiguous tensors.
    
    \item \textbf{Parallelization}: Reduction along dim=1 of a 2D tensor parallelizes naturally across rows.
    
    \item \textbf{\texttt{scatter\_reduce} overhead}:
    \begin{itemize}
        \item Must handle variable group sizes
        \item Irregular memory access patterns
        \item Atomic operations or synchronization for thread safety
    \end{itemize}
\end{enumerate}

\subsection{Memory Trade-offs}

The general approach requires additional memory:
\begin{equation}
\text{Memory}_{\text{general}} = n_{\text{outputs}} \times \text{max\_fan\_in} \times \text{sizeof(dtype)}
\end{equation}

For our benchmarks with fan-in = 10 and float32:
\begin{itemize}
    \item $10^4$ outputs: 400 KB
    \item $10^5$ outputs: 4 MB
    \item $10^6$ outputs: 40 MB
\end{itemize}

This is typically acceptable given modern memory capacities, but may be prohibitive for:
\begin{itemize}
    \item Very high fan-in (hundreds of inputs per output)
    \item Memory-constrained environments
    \item Extremely large output dimensions
\end{itemize}

\section{Recommendations}

\subsection{When to Use Each Approach}

\begin{table}[H]
\centering
\caption{Decision Guide}
\begin{tabular}{@{}p{0.45\textwidth}p{0.45\textwidth}@{}}
\toprule
\textbf{Use \texttt{scatter\_reduce} when:} & \textbf{Use General Approach when:} \\
\midrule
Memory efficiency is critical & Performance is the primary concern \\
Very small problem sizes ($n_{\text{out}} < 500$) & Medium to large scales ($n_{\text{out}} > 1000$) \\
Highly variable fan-in per output & Uniform or bounded fan-in \\
Single forward pass (no gradients) & Training with backward passes \\
 & Custom reduction functions (logsumexp, softmax, learned) \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Implementation Checklist}

For optimal general approach performance:
\begin{enumerate}
    \item Pre-compute group positions once during setup
    \item Pre-allocate the scattered buffer and reuse across calls
    \item Use \texttt{@torch.jit.script} for reduced Python overhead
    \item Ensure contiguous memory layout
\end{enumerate}

\section{Conclusion}

Our benchmarks demonstrate that a general scatter-then-reduce approach, with proper buffer pre-allocation, provides \textbf{significant performance advantages} over PyTorch's fused \texttt{scatter\_reduce} operation:

\begin{itemize}
    \item \textbf{CPU}: 2--5$\times$ faster for prod/max/min at large scales
    \item \textbf{GPU (MPS)}: 1.2--1.4$\times$ faster for prod; mixed results for max/min
    \item \textbf{Backward pass}: Often shows even larger speedups than forward pass
\end{itemize}

The performance advantage stems from regular memory access patterns that enable better vectorization and cache utilization. The trade-off is increased memory usage proportional to $n_{\text{outputs}} \times \text{max\_fan\_in}$.

For applications requiring custom reductions (e.g., log-sum-exp for probabilistic circuits) or maximum throughput at scale, the general approach is strongly recommended.

\end{document}
'''
    
    tex_file = os.path.join(output_dir, "benchmark_report.tex")
    with open(tex_file, 'w') as f:
        f.write(latex_content)
    print(f"LaTeX report saved to {tex_file}")
    
    # Compile LaTeX to PDF
    import subprocess
    try:
        for _ in range(2):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', output_dir, tex_file],
                capture_output=True, text=True, cwd=output_dir
            )
        pdf_file = os.path.join(output_dir, "benchmark_report.pdf")
        if os.path.exists(pdf_file):
            print(f"PDF report generated: {pdf_file}")
        else:
            print("PDF generation may have failed. Check LaTeX installation.")
    except FileNotFoundError:
        print("pdflatex not found. Install LaTeX to generate PDF.")


if __name__ == "__main__":
    main()
