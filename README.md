# Quantum Decision Flow v2.1

A quantum-inspired geometric transformation toolkit that applies Hamiltonian time evolution to classical datasets, producing deformed variants with measurably different topological properties depending on the underlying quantum dynamics.

## Project Structure

```
quantum-decision-flow/
├── src/
│   ├── __init__.py              # Package exports and version
│   ├── deformation.py           # Core quantum deformation engine
│   │   ├── HamiltonianType      # Enum: ZZ_X, HEISENBERG, ISING, XXZ
│   │   ├── create_hamiltonian() # Hamiltonian factory
│   │   ├── make_expectation_circuit()  # Circuit factory (fixes H parameter bug)
│   │   ├── t2_expectations()    # Quantum expectation values
│   │   ├── xy_to_angles()       # Cartesian → spherical encoding
│   │   ├── deform_points()      # Main deformation function
│   │   ├── compute_deformation_field() # Vector field visualization
│   │   └── estimate_topology_preservation() # k-NN metric
│   ├── generator.py             # Dataset generation
│   │   ├── DatasetType          # Enum: MOONS, CIRCLES, SPIRALS, CONCENTRIC_MOONS
│   │   ├── QuantumMoonsConfig   # Configuration dataclass
│   │   ├── generate_classical_dataset()
│   │   ├── generate_quantum_deformed_moons()
│   │   ├── save_dataset()
│   │   └── load_dataset()
│   └── visualization.py         # Plotting utilities
│       ├── plot_moons_grid()
│       ├── plot_deformation_field()
│       ├── plot_deformation_comparison()
│       ├── plot_statistics()
│       └── plot_topology_preservation()
├── tests/
│   ├── __init__.py
│   ├── test_deformation.py      # 28 unit tests
│   ├── test_generator.py        # 18 unit tests
│   └── test_visualization.py    # 8 unit tests
├── notebooks/
│   └── quantum-decision-flow.ipynb
├── requirements.txt             # Pinned dependencies
├── README.md                    # This file
├── CHANGELOG.md                 # Version history and migration guide
├── PHYSICS_NOTES.md             # Theoretical background and experimental results
├── CORRECTIONS.md               # Original bug fixes documentation
├── QUICK_REFERENCE.md           # API cheatsheet
├── run_comparison.py            # Hamiltonian comparison experiment (v2.1)
├── run_spiral_viz.py            # Spiral + Heisenberg visualization script
├── test_periodicity.py          # Extended periodicity hypothesis test
└── test_corrections.py          # Legacy validation script
```

## Installation

```bash
git clone <repo>
cd quantum-decision-flow
pip install -r requirements.txt
```

**Note:** Use `python3` to avoid NumPy/matplotlib version conflicts.

## Quick Start

```python
from src.generator import QuantumMoonsConfig, DatasetType, generate_quantum_deformed_moons
from src.deformation import HamiltonianType
from src.visualization import plot_moons_grid
import matplotlib.pyplot as plt

# Generate spiral data with Heisenberg dynamics
cfg = QuantumMoonsConfig(
    n_samples=400,
    dataset_type=DatasetType.SPIRALS,
    hamiltonian_type=HamiltonianType.HEISENBERG,
    t_values=(0.0, 0.5, 1.0, 1.5, 2.0)
)

X_base, y, X_t, metrics = generate_quantum_deformed_moons(cfg, compute_metrics=True)

# Visualize
fig = plot_moons_grid(X_base, y, X_t)
plt.savefig('output.png', dpi=300)
```

## Key Features

### Multiple Hamiltonians
| Type | Formula | Character |
|------|---------|-----------|
| `ZZ_X` | Z₀Z₁ + X₀ | Non-integrable, Rabi oscillations |
| `HEISENBERG` | X₀X₁ + Y₀Y₁ + Z₀Z₁ | Integrable, SU(2) symmetric |
| `ISING_TRANSVERSE` | Z₀Z₁ + hX₀ + hX₁ | Phase transition at h≈1 |
| `XXZ` | X₀X₁ + Y₀Y₁ + ΔZ₀Z₁ | Tunable anisotropy |

### Multiple Dataset Types
| Type | Description |
|------|-------------|
| `MOONS` | Two interleaving half-circles |
| `CIRCLES` | Concentric circles |
| `SPIRALS` | Interleaving Archimedean spirals |
| `CONCENTRIC_MOONS` | Multiple nested moon pairs |

### Topology Preservation Metrics
Quantify how deformation affects local neighborhood structure:
```python
from src.deformation import estimate_topology_preservation
score = estimate_topology_preservation(X_original, X_deformed, k=5)
# Returns 0-1: 1.0 = perfect preservation
```

## Experimental Finding: Hamiltonian Dynamics Differentiation

Different Hamiltonians produce **measurably different** topology preservation patterns:

| Hamiltonian | Mean | Std | Behavior |
|-------------|------|-----|----------|
| ZZ+X | 0.817 | 0.034 | Higher variance, different phase |
| Heisenberg | 0.828 | 0.025 | More stable, anti-correlated with ZZ+X |

**Correlation between Hamiltonians: -0.064** → Effectively independent dynamics.

See `PHYSICS_NOTES.md` for detailed analysis.

## Physics

### Encoding
```
(x, y) → (θ, φ) → |ψ⟩ = RY(θ)RZ(φ)|0⟩₀ ⊗ RY(θ)RZ(-φ)|0⟩₁
```

### Time Evolution (Trotterization)
```
U(t) ≈ [exp(-iHt/n)]^n
n = max(2, 10|t|)  # Adaptive steps
```

### Deformation
```
x' = x + α·⟨Z₀⟩
y' = y + β·⟨Z₀Z₁⟩ + γ·⟨Z₁⟩
```

## Testing

```bash
python3 -m pytest tests/ -v
# 54 tests, 100% pass rate
```

## Run Hamiltonian Comparison

```bash
python3 run_comparison.py
```

Outputs topology preservation scores for ZZ+X vs Heisenberg across t ∈ [0.5, 5.0].

## Requirements

- Python 3.8+
- PennyLane 0.36-0.39
- NumPy <2.0
- Matplotlib, scikit-learn, scipy

## Version History

- **v2.1.0**: Fixed Hamiltonian parameter bug (circuit factory pattern)
- **v2.0.0**: Multiple Hamiltonians, dataset types, topology metrics
- **v1.0.0**: Initial release

## License

MIT

## Citation

```bibtex
@software{quantum_decision_flow_2_1,
  title  = {Quantum Decision Flow 2.1},
  author = {Altman, Christopher},
  year   = {2025},
  url    = {https://github.com/christopher-altman/quantum-decision-flow}
}
```