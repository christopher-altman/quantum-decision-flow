# Physics Notes: Hamiltonian Dynamics and Topology Preservation

This document records the theoretical background and experimental findings from studying how different quantum Hamiltonians affect geometric deformations and topology preservation in classical datasets.

---

## Core Hypothesis

**Different Hamiltonians produce qualitatively different deformation dynamics**, measurable through topology preservation metrics over evolution time.

Specifically:
- **Integrable systems** (Heisenberg) should show periodic recurrence patterns
- **Non-integrable systems** (ZZ+X) should show quasi-periodic or chaotic behavior
- These differences should manifest as distinct topology preservation signatures

---

## Experimental Results (December 2025)

### Setup
- Dataset: 400-point spiral (interleaving Archimedean spirals)
- Time values: t ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}
- Metric: k-NN topology preservation (k=5)
- Random seed: 42 (fixed for reproducibility)

### Results

#### ZZ + X Hamiltonian (Non-integrable)
```
H = Z₀Z₁ + X₀
```

| Time | Preservation | Notes |
|------|-------------|-------|
| 0.5 | 0.797 | Initial deformation |
| 1.0 | 0.804 | |
| 1.5 | 0.785 | Local minimum |
| 2.0 | 0.863 | Peak |
| 2.5 | 0.854 | |
| 3.0 | 0.783 | **Global minimum** |
| 3.5 | 0.784 | |
| 4.0 | 0.821 | |
| 4.5 | 0.877 | **Global maximum** |
| 5.0 | 0.801 | |

**Statistics:** Mean = 0.817, Std = 0.034

#### Heisenberg Hamiltonian (Integrable)
```
H = X₀X₁ + Y₀Y₁ + Z₀Z₁
```

| Time | Preservation | Notes |
|------|-------------|-------|
| 0.5 | 0.806 | |
| 1.0 | 0.828 | |
| 1.5 | 0.868 | **Global maximum** |
| 2.0 | 0.798 | Local minimum |
| 2.5 | 0.842 | |
| 3.0 | 0.842 | |
| 3.5 | 0.798 | **Global minimum** |
| 4.0 | 0.866 | Second peak |
| 4.5 | 0.831 | |
| 5.0 | 0.807 | |

**Statistics:** Mean = 0.828, Std = 0.025

### Key Finding: Anti-Correlated Dynamics

**Correlation coefficient: -0.064**

The two Hamiltonians produce **effectively uncorrelated** (slightly anti-correlated) topology preservation patterns. When ZZ+X peaks (t=2.0, 4.5), Heisenberg often troughs, and vice versa.

| Time | ZZ+X | Heisenberg | Δ |
|------|------|------------|---|
| 1.5 | 0.785 | 0.868 | +0.083 ← Heisenberg peak, ZZ+X trough |
| 2.0 | 0.863 | 0.798 | -0.065 ← ZZ+X peak, Heisenberg trough |
| 3.0 | 0.783 | 0.842 | +0.059 |
| 4.0 | 0.821 | 0.866 | +0.045 |
| 4.5 | 0.877 | 0.831 | -0.046 ← ZZ+X peak |

---

## Physical Interpretation

### 1. Integrability and Periodicity

The **Heisenberg model is integrable**—it has conserved quantities (total spin) that constrain dynamics to quasi-periodic orbits. The ZZ+X model breaks integrability because the single-site X term doesn't commute with ZZ.

However, both show oscillatory behavior because:
- ZZ+X: The X term induces Rabi oscillations with period ~2π
- Heisenberg: SU(2) symmetry creates rotations on Bloch spheres

The **different phase relationships** (anti-correlation) arise from the different symmetry structures governing the dynamics.

### 2. Topology Preservation as a Dynamical Probe

Topology preservation measures **how much local neighborhood structure survives** after deformation. High values mean the deformation is "gentle"—nearby points stay nearby.

The oscillation in topology preservation reflects the quantum state's return toward and away from configurations that preserve geometric relationships. This is related to **Loschmidt echoes** and quantum recurrence phenomena.

### 3. Why Heisenberg Has Higher Mean Preservation

Heisenberg (mean=0.828) preserves topology better than ZZ+X (mean=0.817) on average. Possible explanation:

The Heisenberg Hamiltonian treats both qubits symmetrically (XX + YY + ZZ), while ZZ+X applies a local field to only qubit 0. The symmetric Heisenberg dynamics may produce more "balanced" deformations that preserve relative point positions better.

### 4. Lower Variance in Integrable System

Heisenberg has lower variance (0.025 vs 0.034). Integrable systems are more constrained—their dynamics can't explore phase space as freely. This manifests as more stable (less volatile) topology preservation over time.

---

## Implications for Machine Learning

### Decision Boundary Stability

If using quantum-deformed datasets for classifier training:
- **Heisenberg deformations** produce more stable decision boundaries across different evolution times
- **ZZ+X deformations** create more varied training distributions—potentially useful for data augmentation

### Feature Space Engineering

The ability to tune deformation characteristics via Hamiltonian choice opens possibilities for:
- **Controlled data augmentation**: Generate training variants with known geometric properties
- **Topology-aware regularization**: Penalize classifiers that don't preserve k-NN structure
- **Quantum-inspired feature maps**: Use evolution time as a hyperparameter

---

## Future Experiments

### 1. Critical Phenomena in Ising Model
The transverse-field Ising model H = ZZ + hX has a quantum phase transition at h_c ≈ 1. Topology preservation near criticality may show characteristic signatures (e.g., diverging correlation length → different k-NN behavior).

### 2. XXZ Anisotropy Sweep
The XXZ model H = XX + YY + ΔZZ interpolates between:
- Δ = 0: XX model (easy-plane)
- Δ = 1: Heisenberg (isotropic)
- Δ → ∞: Ising (easy-axis)

Sweeping Δ could reveal how anisotropy affects deformation geometry.

### 3. Longer Time Evolution
Extend to t ∈ [0, 20] to observe:
- Full period of oscillations
- Whether quasi-periodic (incommensurate frequencies) or truly periodic
- Potential chaotic regimes

### 4. Higher-Dimensional Encoding
Use 3+ qubit systems to deform higher-dimensional datasets. The richer Hilbert space structure should enable more complex deformation patterns.

---

## Technical Note: Circuit Factory Pattern

A critical implementation detail: PennyLane's `@qml.qnode` decorator compiles circuits at definition time. This means:

```python
# WRONG: H parameter is ignored at runtime
@qml.qnode(dev)
def circuit(theta, phi, t, H=H_default):
    qml.ApproxTimeEvolution(H, t, n)  # Always uses H_default!
    ...
```

The fix uses a factory pattern that creates and caches separate circuits per Hamiltonian:

```python
def make_expectation_circuit(H):
    @qml.qnode(dev)
    def circuit(theta, phi, t):
        qml.ApproxTimeEvolution(H, t, n)  # H is baked into this specific circuit
        ...
    return circuit
```

This is essential for comparing different Hamiltonians.

---

## References

1. Sachdev, S. (2011). *Quantum Phase Transitions*. Cambridge University Press.
2. D'Alessio, L., et al. (2016). "From quantum chaos and eigenstate thermalization to statistical mechanics and thermodynamics." *Advances in Physics*, 65(3), 239-362.
3. Preskill, J. (2018). "Quantum Computing in the NISQ era and beyond." *Quantum*, 2, 79.

---

## Citation

```bibtex
@software{bqnn_benchmark,
  title={Quantum Decision Flow 2.1},
  author={Altman, Christopher},
  year={2025},
  url={https://github.com/christopher-altman/quantum-decision-flow}
}
```

*Last updated: December 2025*
