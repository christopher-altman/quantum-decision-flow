from src import (
    QuantumMoonsConfig, 
    DatasetType,
    HamiltonianType,
    generate_quantum_deformed_moons,
    plot_moons_grid,
    estimate_topology_preservation
)

# Configure with Heisenberg dynamics on spiral data
cfg = QuantumMoonsConfig(
    n_samples=400,
    dataset_type=DatasetType.SPIRALS,
    hamiltonian_type=HamiltonianType.HEISENBERG,
    t_values=(0.0, 0.5, 1.0, 1.5, 2.0)
)

# Generate and measure
X_base, y, X_t, metrics = generate_quantum_deformed_moons(cfg, compute_metrics=True)

# Visualize
fig = plot_moons_grid(X_base, y, X_t, show_stats=True)

# Save and display
import matplotlib.pyplot as plt
fig.savefig('spiral_heisenberg_visualization.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'spiral_heisenberg_visualization.png'")
print(f"\nMetrics: {metrics}")
plt.show()
