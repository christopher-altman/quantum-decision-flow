#!/usr/bin/env python3
"""
Hamiltonian Comparison Experiment

Tests whether different Hamiltonians produce distinguishable topology
preservation patterns over time evolution.

Hypothesis: Heisenberg (integrable) vs ZZ+X (non-integrable) should show
different periodicity signatures.
"""

import sys
import time
import numpy as np
sys.path.insert(0, '.')

from src.generator import QuantumMoonsConfig, DatasetType, generate_quantum_deformed_moons
from src.deformation import HamiltonianType, create_hamiltonian, clear_circuit_cache

# Extended time values to capture periodicity
T_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)

def run_experiment(h_type: HamiltonianType, name: str):
    """Run topology preservation experiment for a specific Hamiltonian."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"Hamiltonian: {h_type.value}")
    print(f"{'='*60}")
    
    # Clear cache to ensure fresh circuits
    clear_circuit_cache()
    
    cfg = QuantumMoonsConfig(
        n_samples=400,
        dataset_type=DatasetType.SPIRALS,
        hamiltonian_type=h_type,
        t_values=T_VALUES,
        random_state=42  # Same seed for fair comparison
    )
    
    start = time.time()
    X_base, y, X_t, metrics = generate_quantum_deformed_moons(cfg, compute_metrics=True)
    elapsed = time.time() - start
    
    print(f"\nGeneration time: {elapsed:.1f}s")
    print(f"\nTopology Preservation Scores:")
    print("-" * 40)
    
    results = {}
    for t in sorted(metrics.keys()):
        score = metrics[t]
        results[t] = score
        bar = "█" * int(score * 30)
        print(f"  t={t:4.1f}: {score:.4f} |{bar}")
    
    # Compute statistics
    scores = list(results.values())
    print(f"\n  Mean:   {np.mean(scores):.4f}")
    print(f"  Std:    {np.std(scores):.4f}")
    print(f"  Min:    {np.min(scores):.4f} at t={list(results.keys())[np.argmin(scores)]:.1f}")
    print(f"  Max:    {np.max(scores):.4f} at t={list(results.keys())[np.argmax(scores)]:.1f}")
    
    return results


def main():
    print("=" * 60)
    print("HAMILTONIAN DYNAMICS COMPARISON")
    print("Testing: Heisenberg (integrable) vs ZZ+X (non-integrable)")
    print("=" * 60)
    
    # Run both experiments
    results_zzx = run_experiment(HamiltonianType.ZZ_X, "ZZ + X (Non-integrable)")
    results_heis = run_experiment(HamiltonianType.HEISENBERG, "Heisenberg (Integrable)")
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Time':>6} | {'ZZ+X':>8} | {'Heisenberg':>10} | {'Δ':>8}")
    print("-" * 42)
    
    for t in sorted(results_zzx.keys()):
        zzx = results_zzx[t]
        heis = results_heis[t]
        delta = heis - zzx
        marker = "**" if abs(delta) > 0.02 else ""
        print(f"  {t:4.1f} | {zzx:8.4f} | {heis:10.4f} | {delta:+8.4f} {marker}")
    
    # Check if they're actually different
    zzx_arr = np.array(list(results_zzx.values()))
    heis_arr = np.array(list(results_heis.values()))
    
    if np.allclose(zzx_arr, heis_arr, atol=0.01):
        print("\n⚠️  WARNING: Results are nearly identical!")
        print("    This may indicate the Hamiltonian parameter is not being used.")
    else:
        print(f"\n✓ Hamiltonians produce different dynamics")
        print(f"  Mean absolute difference: {np.mean(np.abs(zzx_arr - heis_arr)):.4f}")
        
        # Correlation analysis
        corr = np.corrcoef(zzx_arr, heis_arr)[0, 1]
        print(f"  Correlation: {corr:.4f}")
        
        if corr < 0.7:
            print("  → Low correlation suggests qualitatively different behavior")
        else:
            print("  → High correlation suggests similar periodic structure")


if __name__ == "__main__":
    main()
