from src import (
    QuantumMoonsConfig, 
    DatasetType,
    HamiltonianType,
    generate_quantum_deformed_moons,
    plot_moons_grid,
    estimate_topology_preservation
)

# Test the periodicity hypothesis
cfg_extended = QuantumMoonsConfig(
    n_samples=400,
    dataset_type=DatasetType.SPIRALS,
    hamiltonian_type=HamiltonianType.HEISENBERG,
    t_values=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
)

# Compare with ZZ_X (should show monotonic decay)
cfg_zzx = QuantumMoonsConfig(
    n_samples=400,
    dataset_type=DatasetType.SPIRALS,
    hamiltonian_type=HamiltonianType.ZZ_X,
    t_values=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
)

# Generate and measure
X_base_ext, y_ext, X_t_ext, metrics_ext = generate_quantum_deformed_moons(cfg_extended, compute_metrics=True)
X_base_zzx, y_zzx, X_t_zzx, metrics_zzx = generate_quantum_deformed_moons(cfg_zzx, compute_metrics=True)

# Visualize
import matplotlib.pyplot as plt

fig_ext = plot_moons_grid(X_base_ext, y_ext, X_t_ext, show_stats=True)
fig_ext.savefig('heisenberg_extended_periodicity.png', dpi=300, bbox_inches='tight')
print("Heisenberg figure saved as 'heisenberg_extended_periodicity.png'")
print(f"\nHeisenberg Metrics: {metrics_ext}")

fig_zzx = plot_moons_grid(X_base_zzx, y_zzx, X_t_zzx, show_stats=True)
fig_zzx.savefig('zzx_extended_comparison.png', dpi=300, bbox_inches='tight')
print("ZZ_X figure saved as 'zzx_extended_comparison.png'")
print(f"\nZZ_X Metrics: {metrics_zzx}")

plt.show()
