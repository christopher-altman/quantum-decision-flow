"""
Dataset generator for quantum-deformed concentric moons.

Provides utilities to generate classical moon-shaped datasets and apply
quantum-inspired geometric deformations parameterized by evolution time.

The generator supports:
- Configurable sample sizes, noise levels, and time evolution parameters
- Multiple dataset variants (moons, circles, spirals)
- Batch generation with progress tracking
- Reproducible random states
"""

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple, Optional, List
from enum import Enum
import numpy as np
from sklearn.datasets import make_moons, make_circles

from .deformation import (
    deform_points, 
    HamiltonianType, 
    create_hamiltonian,
    estimate_topology_preservation
)


class DatasetType(Enum):
    """Available base dataset types."""
    MOONS = "moons"
    CIRCLES = "circles"
    SPIRALS = "spirals"
    CONCENTRIC_MOONS = "concentric_moons"


@dataclass
class QuantumMoonsConfig:
    """
    Configuration for quantum moon dataset generation.
    
    Attributes:
        n_samples: Total number of points (split evenly between classes)
        noise: Standard deviation of Gaussian noise (0 = no noise)
        random_state: Random seed for reproducibility (None = random)
        t_values: Time evolution parameters for deformation
        dataset_type: Type of base dataset to generate
        hamiltonian_type: Hamiltonian for quantum evolution
        alpha: X-deformation weight
        beta: Y-deformation weight (ZZ term)
        gamma: Y-deformation weight (Z1 term)
    """
    n_samples: int = 400
    noise: float = 0.08
    random_state: Optional[int] = 42
    t_values: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0)
    dataset_type: DatasetType = DatasetType.MOONS
    hamiltonian_type: HamiltonianType = HamiltonianType.ZZ_X
    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        if self.noise < 0:
            raise ValueError(f"noise must be non-negative, got {self.noise}")
        if not self.t_values:
            raise ValueError("t_values must contain at least one value")
        if not isinstance(self.dataset_type, DatasetType):
            self.dataset_type = DatasetType(self.dataset_type)
        if not isinstance(self.hamiltonian_type, HamiltonianType):
            self.hamiltonian_type = HamiltonianType(self.hamiltonian_type)


def generate_spirals(
    n_samples: int,
    noise: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate interleaving spiral dataset.
    
    Creates two Archimedean spirals that interleave, providing a more
    challenging classification geometry than moons.
    
    Args:
        n_samples: Total points (split between spirals)
        noise: Gaussian noise standard deviation
        random_state: Random seed
        
    Returns:
        (X, y) tuple of points and labels
    """
    rng = np.random.default_rng(random_state)
    n_per_class = n_samples // 2
    
    # First spiral
    theta1 = np.sqrt(rng.uniform(0, 1, n_per_class)) * 2 * np.pi
    r1 = theta1 / (2 * np.pi)
    x1 = r1 * np.cos(theta1) + noise * rng.standard_normal(n_per_class)
    y1 = r1 * np.sin(theta1) + noise * rng.standard_normal(n_per_class)
    
    # Second spiral (rotated 180°)
    theta2 = np.sqrt(rng.uniform(0, 1, n_per_class)) * 2 * np.pi
    r2 = theta2 / (2 * np.pi)
    x2 = -r2 * np.cos(theta2) + noise * rng.standard_normal(n_per_class)
    y2 = -r2 * np.sin(theta2) + noise * rng.standard_normal(n_per_class)
    
    X = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    
    return X.astype(np.float32), y.astype(np.int64)


def generate_concentric_moons(
    n_samples: int,
    noise: float = 0.1,
    random_state: Optional[int] = None,
    n_rings: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate multiple concentric moon rings.
    
    Extends standard moons to multiple concentric layers, useful for
    studying how quantum deformation affects nested structures.
    
    Args:
        n_samples: Total points (distributed across rings)
        noise: Gaussian noise standard deviation
        random_state: Random seed
        n_rings: Number of concentric moon pairs
        
    Returns:
        (X, y) tuple of points and labels
    """
    rng = np.random.default_rng(random_state)
    n_per_ring = n_samples // n_rings
    
    all_X = []
    all_y = []
    
    for ring in range(n_rings):
        X_ring, y_ring = make_moons(
            n_samples=n_per_ring,
            noise=noise,
            random_state=rng.integers(0, 2**31) if random_state else None
        )
        # Scale and shift each ring
        scale = 1.0 + 0.8 * ring
        X_ring = X_ring * scale
        all_X.append(X_ring)
        all_y.append(y_ring + 2 * ring)  # Different labels per ring
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    return X.astype(np.float32), y.astype(np.int64)


def generate_classical_dataset(cfg: QuantumMoonsConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate classical dataset based on configuration.
    
    Dispatches to appropriate generator based on dataset_type.
    
    Args:
        cfg: Configuration object
        
    Returns:
        (X, y) tuple of points and labels
    """
    if cfg.dataset_type == DatasetType.MOONS:
        X, y = make_moons(
            n_samples=cfg.n_samples,
            noise=cfg.noise,
            random_state=cfg.random_state,
        )
    elif cfg.dataset_type == DatasetType.CIRCLES:
        X, y = make_circles(
            n_samples=cfg.n_samples,
            noise=cfg.noise,
            random_state=cfg.random_state,
            factor=0.5
        )
    elif cfg.dataset_type == DatasetType.SPIRALS:
        X, y = generate_spirals(
            n_samples=cfg.n_samples,
            noise=cfg.noise,
            random_state=cfg.random_state
        )
    elif cfg.dataset_type == DatasetType.CONCENTRIC_MOONS:
        X, y = generate_concentric_moons(
            n_samples=cfg.n_samples,
            noise=cfg.noise,
            random_state=cfg.random_state
        )
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset_type}")
    
    return X.astype(np.float32), y.astype(np.int64)


# Backwards compatibility alias
def generate_classical_moons(cfg: QuantumMoonsConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Generate classical moons dataset (backwards compatible)."""
    return generate_classical_dataset(cfg)


def generate_quantum_deformed_moons(
    cfg: QuantumMoonsConfig,
    compute_metrics: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[float, np.ndarray], Optional[Dict[float, float]]]:
    """
    Generate classical dataset and quantum-deformed variants.
    
    Creates base dataset, then applies quantum deformation at each
    specified time value to study geometric evolution.
    
    Args:
        cfg: Configuration specifying all generation parameters
        compute_metrics: If True, compute topology preservation metrics
        
    Returns:
        Tuple of:
            X_base: Base classical dataset (n_samples, 2)
            y: Labels (n_samples,)
            X_t: Dict mapping t → deformed dataset
            metrics: Dict mapping t → topology score (if compute_metrics)
    """
    X_base, y = generate_classical_dataset(cfg)
    X_t: Dict[float, np.ndarray] = {}
    metrics: Dict[float, float] = {} if compute_metrics else None
    
    H = create_hamiltonian(cfg.hamiltonian_type)
    
    for t in cfg.t_values:
        X_t[t] = deform_points(
            X_base, 
            t=t, 
            alpha=cfg.alpha, 
            beta=cfg.beta,
            gamma=cfg.gamma,
            H=H
        )
        
        if compute_metrics and t != 0:
            metrics[t] = estimate_topology_preservation(X_base, X_t[t])
    
    if compute_metrics:
        return X_base, y, X_t, metrics
    return X_base, y, X_t, None


def save_dataset(
    filename: str,
    X_base: np.ndarray,
    y: np.ndarray,
    X_t: Dict[float, np.ndarray],
    cfg: Optional[QuantumMoonsConfig] = None,
    metrics: Optional[Dict[float, float]] = None
) -> None:
    """
    Save generated dataset to compressed NPZ file.
    
    Args:
        filename: Output file path
        X_base: Base dataset
        y: Labels
        X_t: Time-evolved variants
        cfg: Optional config for metadata
        metrics: Optional topology metrics
    """
    save_dict = {
        'X_base': X_base,
        'y': y,
        't_values': np.array(sorted(X_t.keys())),
    }
    
    for t, X in X_t.items():
        save_dict[f'X_t_{t:.2f}'] = X
    
    if cfg is not None:
        save_dict['config'] = np.array([
            cfg.n_samples, cfg.noise, 
            cfg.random_state or -1,
            cfg.alpha, cfg.beta, cfg.gamma
        ])
    
    if metrics is not None:
        save_dict['topology_scores'] = np.array([
            [t, score] for t, score in sorted(metrics.items())
        ])
    
    np.savez_compressed(filename, **save_dict)


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
    """
    Load dataset from NPZ file.
    
    Args:
        filename: Input file path
        
    Returns:
        (X_base, y, X_t) tuple
    """
    data = np.load(filename)
    X_base = data['X_base']
    y = data['y']
    t_values = data['t_values']
    
    X_t = {}
    for t in t_values:
        key = f'X_t_{t:.2f}'
        if key in data:
            X_t[t] = data[key]
    
    return X_base, y, X_t


if __name__ == "__main__":
    cfg = QuantumMoonsConfig()
    X_base, y, X_t, metrics = generate_quantum_deformed_moons(cfg, compute_metrics=True)
    
    save_dataset(
        "quantum_moons_day02.npz",
        X_base, y, X_t, cfg, metrics
    )
    
    print(f"Saved quantum_moons_day02.npz")
    print(f"  Samples: {len(X_base)}")
    print(f"  Time variants: {list(X_t.keys())}")
    if metrics:
        print("  Topology preservation:")
        for t, score in sorted(metrics.items()):
            print(f"    t={t:.1f}: {score:.3f}")
