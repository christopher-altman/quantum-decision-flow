"""
Quantum deformation of classical 2D datasets via Hamiltonian time evolution.

This module implements quantum-inspired geometric transformations where classical
2D points are encoded into quantum states, evolved under configurable Hamiltonians,
and decoded back to deformed coordinates.

Physics:
    The default Hamiltonian H = ZZ + X combines:
    - ZZ interaction: Creates entanglement-driven correlations (y-deformation)
    - X field: Introduces transverse local deformations (x-deformation)
    
    Time evolution is approximated using Trotterization:
    U(t) ≈ [exp(-iH·t/n)]^n
    
    Adaptive step sizing ensures error scales as O(t²/n), maintaining accuracy
    for large evolution times.

IMPORTANT: Different Hamiltonians produce qualitatively different dynamics:
    - ZZ+X: Non-integrable, quasi-periodic with X-field Rabi oscillations
    - Heisenberg (XX+YY+ZZ): Integrable, SU(2) symmetric, different periodicity
    - Ising: Phase transition behavior at critical field strengths
    - XXZ: Tunable anisotropy interpolates between regimes
"""

from typing import Tuple, Optional, Callable, Dict
from enum import Enum
import pennylane as qml
from pennylane import numpy as np
import functools


class HamiltonianType(Enum):
    """Available Hamiltonian configurations for quantum deformation."""
    ZZ_X = "zz_x"               # Default: ZZ + X (entanglement + transverse)
    HEISENBERG = "heisenberg"   # XX + YY + ZZ (isotropic exchange)
    ISING_TRANSVERSE = "ising"  # ZZ + hX (tunable transverse field)
    XXZ = "xxz"                 # XX + YY + ΔZZ (anisotropic Heisenberg)


def create_hamiltonian(
    h_type: HamiltonianType = HamiltonianType.ZZ_X,
    coupling: float = 1.0,
    field: float = 1.0,
    anisotropy: float = 1.0
) -> qml.Hamiltonian:
    """
    Create a 2-qubit Hamiltonian for quantum deformation.
    
    Args:
        h_type: Type of Hamiltonian to construct
        coupling: Strength of qubit-qubit interaction
        field: Strength of local field terms
        anisotropy: Anisotropy parameter (for XXZ model)
        
    Returns:
        PennyLane Hamiltonian object
    """
    if h_type == HamiltonianType.ZZ_X:
        return qml.Hamiltonian(
            [coupling, field],
            [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0)]
        )
    elif h_type == HamiltonianType.HEISENBERG:
        return qml.Hamiltonian(
            [coupling, coupling, coupling],
            [
                qml.PauliX(0) @ qml.PauliX(1),
                qml.PauliY(0) @ qml.PauliY(1),
                qml.PauliZ(0) @ qml.PauliZ(1)
            ]
        )
    elif h_type == HamiltonianType.ISING_TRANSVERSE:
        return qml.Hamiltonian(
            [coupling, field, field],
            [
                qml.PauliZ(0) @ qml.PauliZ(1),
                qml.PauliX(0),
                qml.PauliX(1)
            ]
        )
    elif h_type == HamiltonianType.XXZ:
        return qml.Hamiltonian(
            [coupling, coupling, coupling * anisotropy],
            [
                qml.PauliX(0) @ qml.PauliX(1),
                qml.PauliY(0) @ qml.PauliY(1),
                qml.PauliZ(0) @ qml.PauliZ(1)
            ]
        )
    else:
        raise ValueError(f"Unknown Hamiltonian type: {h_type}")


# Default Hamiltonian: H = ZZ + X
H_DEFAULT = create_hamiltonian(HamiltonianType.ZZ_X)

# Global device (cached for performance)
_DEVICE_CACHE = {}

def get_device(wires: int = 2) -> qml.Device:
    """Get or create a cached quantum device."""
    if wires not in _DEVICE_CACHE:
        _DEVICE_CACHE[wires] = qml.device("default.qubit", wires=wires)
    return _DEVICE_CACHE[wires]


DEV = get_device(2)


# ============================================================================
# HAMILTONIAN-AWARE CIRCUIT FACTORY
# ============================================================================
# 
# PennyLane's @qml.qnode decorator compiles circuits at definition time.
# To support runtime Hamiltonian selection, we use a factory pattern that
# creates and caches separate qnodes per Hamiltonian.
# ============================================================================

_CIRCUIT_CACHE: Dict[str, Callable] = {}


def _hamiltonian_key(H: qml.Hamiltonian) -> str:
    """Generate a unique key for a Hamiltonian for caching."""
    coeffs = tuple(float(c) for c in H.coeffs)
    ops = tuple(str(op) for op in H.ops)
    return str((coeffs, ops))


def make_expectation_circuit(H: qml.Hamiltonian) -> Callable:
    """
    Factory function to create Hamiltonian-specific quantum circuits.
    
    PennyLane qnodes compile at definition time, so runtime Hamiltonian
    parameters are not respected. This factory creates and caches separate
    circuits for each unique Hamiltonian.
    
    Args:
        H: The Hamiltonian for time evolution
        
    Returns:
        A callable qnode that computes (Z0, Z1, ZZ) expectations
    """
    key = _hamiltonian_key(H)
    
    if key in _CIRCUIT_CACHE:
        return _CIRCUIT_CACHE[key]
    
    @qml.qnode(DEV)
    def circuit(theta: float, phi: float, t: float) -> Tuple[float, float, float]:
        qml.RY(theta, wires=0)
        qml.RZ(phi, wires=0)
        qml.RY(theta, wires=1)
        qml.RZ(-phi, wires=1)
        
        n_steps = max(2, int(10 * abs(t))) if t != 0 else 1
        if t != 0:
            qml.ApproxTimeEvolution(H, t, n_steps)
        
        z0 = qml.expval(qml.PauliZ(0))
        z1 = qml.expval(qml.PauliZ(1))
        zz = qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        return z0, z1, zz
    
    _CIRCUIT_CACHE[key] = circuit
    return circuit


def t2_expectations(
    theta: float, 
    phi: float, 
    t: float, 
    H: Optional[qml.Hamiltonian] = None
) -> Tuple[float, float, float]:
    """
    Compute expectation values after Hamiltonian time evolution.
    """
    if H is None:
        H = H_DEFAULT
    
    circuit = make_expectation_circuit(H)
    return circuit(theta, phi, t)


def xy_to_angles(x: float, y: float, scale: float = 1.0) -> Tuple[float, float]:
    """Map 2D Cartesian (x, y) to spherical angles (θ, φ)."""
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    sigmoid_r = 1.0 / (1.0 + np.exp(-r / scale))
    theta = np.pi * sigmoid_r
    return float(theta), float(phi)


def deform_points(
    X: np.ndarray,
    t: float,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.0,
    H: Optional[qml.Hamiltonian] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> np.ndarray:
    """Apply quantum deformation to 2D points via Hamiltonian evolution."""
    X = np.array(X)
    
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must have shape (n_samples, 2), got {X.shape}")
    
    if len(X) == 0:
        return X.copy()
    
    if H is None:
        H = H_DEFAULT
    
    X_def = np.zeros_like(X, dtype=np.float64)
    n_points = len(X)

    for i, (x, y) in enumerate(X):
        theta, phi = xy_to_angles(float(x), float(y))
        z0, z1, zz = t2_expectations(theta, phi, t, H=H)
        
        z0, z1, zz = float(z0), float(z1), float(zz)
        
        X_def[i, 0] = float(x) + alpha * z0
        X_def[i, 1] = float(y) + beta * zz + gamma * z1
        
        if progress_callback is not None:
            progress_callback(i + 1, n_points)

    return X_def


def compute_deformation_field(
    grid_size: int = 20,
    x_range: Tuple[float, float] = (-2, 3),
    y_range: Tuple[float, float] = (-1.5, 2),
    t: float = 1.0,
    alpha: float = 0.4,
    beta: float = 0.4,
    H: Optional[qml.Hamiltonian] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the deformation vector field over a 2D grid."""
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    deformed = deform_points(points, t, alpha, beta, H=H)
    
    U = (deformed[:, 0] - points[:, 0]).reshape(grid_size, grid_size)
    V = (deformed[:, 1] - points[:, 1]).reshape(grid_size, grid_size)
    
    return X_grid, Y_grid, U, V


def estimate_topology_preservation(
    X_original: np.ndarray,
    X_deformed: np.ndarray,
    k: int = 5
) -> float:
    """Estimate topology preservation via k-nearest neighbor consistency."""
    from scipy.spatial import KDTree
    
    if len(X_original) <= k:
        return 1.0
    
    tree_orig = KDTree(X_original)
    tree_def = KDTree(X_deformed)
    
    _, idx_orig = tree_orig.query(X_original, k=k+1)
    _, idx_def = tree_def.query(X_deformed, k=k+1)
    
    idx_orig = idx_orig[:, 1:]
    idx_def = idx_def[:, 1:]
    
    preserved = 0
    for i in range(len(X_original)):
        preserved += len(set(idx_orig[i]) & set(idx_def[i]))
    
    return preserved / (len(X_original) * k)


def clear_circuit_cache():
    """Clear the cached quantum circuits."""
    global _CIRCUIT_CACHE
    _CIRCUIT_CACHE = {}
