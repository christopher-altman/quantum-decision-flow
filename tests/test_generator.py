"""
Unit tests for dataset generator module.

Tests cover:
- Configuration validation
- Dataset generation correctness
- Multiple dataset types
- Save/load functionality
"""

import pytest
import numpy as np
import tempfile
import os
import sys
sys.path.insert(0, '..')


class TestQuantumMoonsConfig:
    """Tests for configuration dataclass."""
    
    def test_default_values(self):
        """Default configuration should be valid."""
        from src.generator import QuantumMoonsConfig
        cfg = QuantumMoonsConfig()
        assert cfg.n_samples == 400
        assert cfg.noise == 0.08
        assert cfg.random_state == 42
    
    def test_negative_samples_rejected(self):
        """Negative n_samples should raise ValueError."""
        from src.generator import QuantumMoonsConfig
        with pytest.raises(ValueError):
            QuantumMoonsConfig(n_samples=-1)
    
    def test_zero_samples_rejected(self):
        """Zero n_samples should raise ValueError."""
        from src.generator import QuantumMoonsConfig
        with pytest.raises(ValueError):
            QuantumMoonsConfig(n_samples=0)
    
    def test_negative_noise_rejected(self):
        """Negative noise should raise ValueError."""
        from src.generator import QuantumMoonsConfig
        with pytest.raises(ValueError):
            QuantumMoonsConfig(noise=-0.1)
    
    def test_zero_noise_valid(self):
        """Zero noise should be valid."""
        from src.generator import QuantumMoonsConfig
        cfg = QuantumMoonsConfig(noise=0.0)
        assert cfg.noise == 0.0
    
    def test_empty_t_values_rejected(self):
        """Empty t_values should raise ValueError."""
        from src.generator import QuantumMoonsConfig
        with pytest.raises(ValueError):
            QuantumMoonsConfig(t_values=())
    
    def test_custom_t_values(self):
        """Custom t_values should be accepted."""
        from src.generator import QuantumMoonsConfig
        cfg = QuantumMoonsConfig(t_values=(0.0, 0.25, 0.5))
        assert len(cfg.t_values) == 3


class TestGenerateClassicalDataset:
    """Tests for classical dataset generation."""
    
    def test_moons_shape(self):
        """Generated moons should have correct shape."""
        from src.generator import QuantumMoonsConfig, generate_classical_dataset
        cfg = QuantumMoonsConfig(n_samples=100)
        X, y = generate_classical_dataset(cfg)
        assert X.shape == (100, 2)
        assert y.shape == (100,)
    
    def test_moons_labels(self):
        """Moons should have binary labels."""
        from src.generator import QuantumMoonsConfig, generate_classical_dataset
        cfg = QuantumMoonsConfig(n_samples=100)
        X, y = generate_classical_dataset(cfg)
        assert set(np.unique(y)) == {0, 1}
    
    def test_reproducibility(self):
        """Same random_state should produce same results."""
        from src.generator import QuantumMoonsConfig, generate_classical_dataset
        cfg1 = QuantumMoonsConfig(n_samples=50, random_state=123)
        cfg2 = QuantumMoonsConfig(n_samples=50, random_state=123)
        X1, y1 = generate_classical_dataset(cfg1)
        X2, y2 = generate_classical_dataset(cfg2)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_different_seeds_different_results(self):
        """Different seeds should produce different results."""
        from src.generator import QuantumMoonsConfig, generate_classical_dataset
        cfg1 = QuantumMoonsConfig(n_samples=50, random_state=1)
        cfg2 = QuantumMoonsConfig(n_samples=50, random_state=2)
        X1, _ = generate_classical_dataset(cfg1)
        X2, _ = generate_classical_dataset(cfg2)
        assert not np.allclose(X1, X2)
    
    def test_circles_dataset(self):
        """Circles dataset should work."""
        from src.generator import QuantumMoonsConfig, generate_classical_dataset, DatasetType
        cfg = QuantumMoonsConfig(n_samples=100, dataset_type=DatasetType.CIRCLES)
        X, y = generate_classical_dataset(cfg)
        assert X.shape == (100, 2)
        assert len(np.unique(y)) == 2
    
    def test_spirals_dataset(self):
        """Spirals dataset should work."""
        from src.generator import QuantumMoonsConfig, generate_classical_dataset, DatasetType
        cfg = QuantumMoonsConfig(n_samples=100, dataset_type=DatasetType.SPIRALS)
        X, y = generate_classical_dataset(cfg)
        assert X.shape == (100, 2)
        assert len(np.unique(y)) == 2


class TestGenerateQuantumDeformedMoons:
    """Tests for quantum-deformed dataset generation."""
    
    def test_output_structure(self):
        """Output should have correct structure."""
        from src.generator import QuantumMoonsConfig, generate_quantum_deformed_moons
        cfg = QuantumMoonsConfig(n_samples=50, t_values=(0.0, 0.5))
        X_base, y, X_t, metrics = generate_quantum_deformed_moons(cfg)
        
        assert X_base.shape == (50, 2)
        assert y.shape == (50,)
        assert len(X_t) == 2
        assert 0.0 in X_t
        assert 0.5 in X_t
    
    def test_deformed_shapes(self):
        """All deformed variants should have same shape."""
        from src.generator import QuantumMoonsConfig, generate_quantum_deformed_moons
        cfg = QuantumMoonsConfig(n_samples=50, t_values=(0.0, 0.5, 1.0))
        X_base, y, X_t, _ = generate_quantum_deformed_moons(cfg)
        
        for t, X_def in X_t.items():
            assert X_def.shape == X_base.shape
    
    def test_nonzero_deformation(self):
        """Non-zero t should produce different points."""
        from src.generator import QuantumMoonsConfig, generate_quantum_deformed_moons
        cfg = QuantumMoonsConfig(n_samples=50, t_values=(0.0, 1.0))
        X_base, y, X_t, _ = generate_quantum_deformed_moons(cfg)
        
        assert not np.allclose(X_base, X_t[1.0])
    
    def test_metrics_computation(self):
        """Metrics should be computed when requested."""
        from src.generator import QuantumMoonsConfig, generate_quantum_deformed_moons
        cfg = QuantumMoonsConfig(n_samples=50, t_values=(0.0, 0.5))
        _, _, _, metrics = generate_quantum_deformed_moons(cfg, compute_metrics=True)
        
        assert metrics is not None
        assert 0.5 in metrics
        assert 0 <= metrics[0.5] <= 1


class TestSaveLoad:
    """Tests for save/load functionality."""
    
    def test_save_load_roundtrip(self):
        """Data should survive save/load cycle."""
        from src.generator import (
            QuantumMoonsConfig, generate_quantum_deformed_moons,
            save_dataset, load_dataset
        )
        
        cfg = QuantumMoonsConfig(n_samples=50, t_values=(0.0, 0.5))
        X_base, y, X_t, _ = generate_quantum_deformed_moons(cfg)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filename = f.name
        
        try:
            save_dataset(filename, X_base, y, X_t, cfg)
            X_base_loaded, y_loaded, X_t_loaded = load_dataset(filename)
            
            np.testing.assert_array_almost_equal(X_base, X_base_loaded)
            np.testing.assert_array_equal(y, y_loaded)
            assert len(X_t) == len(X_t_loaded)
        finally:
            os.unlink(filename)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
