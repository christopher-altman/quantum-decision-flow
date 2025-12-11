"""
Quantum Decision Flow v2.1

A quantum-inspired geometric transformation toolkit that applies
Hamiltonian time evolution to classical datasets, creating deformed
variants for studying quantum effects on decision boundaries.

Modules:
    generator: Dataset generation and configuration
    deformation: Quantum deformation via 2-qubit Hamiltonian evolution
    visualization: Plotting utilities for comparative visualization

New in v2.0:
    - Multiple Hamiltonian types (ZZ_X, Heisenberg, Ising, XXZ)
    - Multiple dataset types (moons, circles, spirals, concentric_moons)
    - Topology preservation metrics
    - Deformation vector field computation
    - Enhanced visualization functions
"""

__version__ = "2.1"
__author__ = "Christopher"

from .deformation import (
    HamiltonianType,
    create_hamiltonian,
    deform_points,
    xy_to_angles,
    t2_expectations,
    compute_deformation_field,
    estimate_topology_preservation,
)

from .generator import (
    DatasetType,
    QuantumMoonsConfig,
    generate_classical_moons,
    generate_classical_dataset,
    generate_quantum_deformed_moons,
    save_dataset,
    load_dataset,
)

from .visualization import (
    plot_moons_grid,
    plot_deformation_field,
    plot_deformation_comparison,
    plot_statistics,
    plot_topology_preservation,
)

__all__ = [
    # Version
    "__version__",
    # Deformation
    "HamiltonianType",
    "create_hamiltonian",
    "deform_points",
    "xy_to_angles",
    "t2_expectations",
    "compute_deformation_field",
    "estimate_topology_preservation",
    # Generator
    "DatasetType",
    "QuantumMoonsConfig",
    "generate_classical_moons",
    "generate_classical_dataset",
    "generate_quantum_deformed_moons",
    "save_dataset",
    "load_dataset",
    # Visualization
    "plot_moons_grid",
    "plot_deformation_field",
    "plot_deformation_comparison",
    "plot_statistics",
    "plot_topology_preservation",
]
