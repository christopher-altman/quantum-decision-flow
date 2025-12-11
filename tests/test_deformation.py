"""
Unit tests for quantum deformation module.

Tests cover:
- Angle mapping functions
- Quantum circuit execution
- Deformation correctness
- Edge cases and error handling
- Numerical accuracy
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')


class TestXYToAngles:
    """Tests for xy_to_angles function."""
    
    def test_origin(self):
        """Origin should map to theta=pi/2 (sigmoid of 0)."""
        from src.deformation import xy_to_angles
        theta, phi = xy_to_angles(0.0, 0.0)
        assert np.isclose(theta, np.pi / 2, atol=1e-6)
        assert np.isclose(phi, 0.0, atol=1e-6)
    
    def test_positive_x_axis(self):
        """Point on positive x-axis should have phi=0."""
        from src.deformation import xy_to_angles
        theta, phi = xy_to_angles(1.0, 0.0)
        assert np.isclose(phi, 0.0, atol=1e-6)
        assert 0 < theta < np.pi  # Within valid range
    
    def test_positive_y_axis(self):
        """Point on positive y-axis should have phi=pi/2."""
        from src.deformation import xy_to_angles
        theta, phi = xy_to_angles(0.0, 1.0)
        assert np.isclose(phi, np.pi / 2, atol=1e-6)
    
    def test_negative_x_axis(self):
        """Point on negative x-axis should have phi=±pi."""
        from src.deformation import xy_to_angles
        theta, phi = xy_to_angles(-1.0, 0.0)
        assert np.isclose(abs(phi), np.pi, atol=1e-6)
    
    def test_theta_bounds(self):
        """Theta should always be in [0, pi]."""
        from src.deformation import xy_to_angles
        test_points = [(1e-10, 1e-10), (1e6, 1e6), (-100, 50), (0.5, -0.3)]
        for x, y in test_points:
            theta, phi = xy_to_angles(x, y)
            assert 0 <= theta <= np.pi, f"theta={theta} out of bounds for ({x}, {y})"
    
    def test_phi_bounds(self):
        """Phi should always be in [-pi, pi]."""
        from src.deformation import xy_to_angles
        test_points = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        for x, y in test_points:
            theta, phi = xy_to_angles(x, y)
            assert -np.pi <= phi <= np.pi, f"phi={phi} out of bounds"
    
    def test_large_values(self):
        """Large values should saturate theta near pi."""
        from src.deformation import xy_to_angles
        theta, _ = xy_to_angles(1e6, 1e6)
        assert theta > 0.99 * np.pi  # Should be very close to pi


class TestT2Expectations:
    """Tests for quantum circuit expectation values."""
    
    def test_zero_time_identity(self):
        """Zero evolution time should return initial state expectations."""
        from src.deformation import t2_expectations
        z0, z1, zz = t2_expectations(np.pi/2, 0.0, t=0.0)
        # At theta=pi/2, initial state is |+⟩, so ⟨Z⟩ ≈ 0
        assert abs(float(z0)) < 0.1
        assert abs(float(z1)) < 0.1
    
    def test_expectation_bounds(self):
        """All expectations should be in [-1, 1]."""
        from src.deformation import t2_expectations
        for t in [0.0, 0.5, 1.0, 2.0]:
            for theta in [0.0, np.pi/4, np.pi/2, np.pi]:
                for phi in [0.0, np.pi/2, np.pi]:
                    z0, z1, zz = t2_expectations(theta, phi, t)
                    assert -1 <= float(z0) <= 1, f"z0={z0} out of bounds"
                    assert -1 <= float(z1) <= 1, f"z1={z1} out of bounds"
                    assert -1 <= float(zz) <= 1, f"zz={zz} out of bounds"
    
    def test_negative_time(self):
        """Negative time should also produce valid results."""
        from src.deformation import t2_expectations
        z0, z1, zz = t2_expectations(1.0, 0.5, t=-1.0)
        assert -1 <= float(z0) <= 1
        assert -1 <= float(zz) <= 1
    
    def test_symmetry(self):
        """Phi sign flip should preserve z0 but may change correlations."""
        from src.deformation import t2_expectations
        z0_pos, z1_pos, zz_pos = t2_expectations(1.0, 0.5, 1.0)
        z0_neg, z1_neg, zz_neg = t2_expectations(1.0, -0.5, 1.0)
        # Z0 expectation should be similar (same theta)
        assert np.isclose(float(z0_pos), float(z0_neg), atol=0.1)


class TestDeformPoints:
    """Tests for main deformation function."""
    
    def test_shape_preservation(self):
        """Output should have same shape as input."""
        from src.deformation import deform_points
        X = np.random.randn(10, 2)
        X_def = deform_points(X, t=0.5)
        assert X_def.shape == X.shape
    
    def test_empty_input(self):
        """Empty array should return empty array."""
        from src.deformation import deform_points
        X = np.array([]).reshape(0, 2)
        X_def = deform_points(X, t=1.0)
        assert X_def.shape == (0, 2)
    
    def test_single_point(self):
        """Single point should work."""
        from src.deformation import deform_points
        X = np.array([[1.0, 0.5]])
        X_def = deform_points(X, t=0.5)
        assert X_def.shape == (1, 2)
    
    def test_zero_time_identity(self):
        """t=0 should return original points (approximately)."""
        from src.deformation import deform_points
        X = np.array([[1.0, 0.5], [0.0, 1.0]])
        X_def = deform_points(X, t=0.0)
        # With t=0, deformation should be minimal
        diff = np.abs(X_def - X).max()
        assert diff < 0.5  # Some numerical tolerance
    
    def test_1d_input_reshaped(self):
        """1D array with 2 elements should be auto-reshaped to (1, 2)."""
        from src.deformation import deform_points
        X = np.array([1.0, 2.0])
        X_def = deform_points(X, t=0.5)
        assert X_def.shape == (1, 2)
    
    def test_invalid_1d_wrong_size(self):
        """1D array with wrong size should raise ValueError."""
        from src.deformation import deform_points
        X = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            deform_points(X, t=1.0)
    
    def test_invalid_3d_points(self):
        """3D points should raise ValueError."""
        from src.deformation import deform_points
        X = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError):
            deform_points(X, t=1.0)
    
    def test_deformation_increases_with_time(self):
        """Larger time should generally produce more deformation."""
        from src.deformation import deform_points
        X = np.array([[0.5, 0.3], [1.0, -0.2], [0.0, 0.8]])
        
        X_t05 = deform_points(X, t=0.5)
        X_t10 = deform_points(X, t=1.0)
        X_t20 = deform_points(X, t=2.0)
        
        diff_05 = np.linalg.norm(X_t05 - X)
        diff_10 = np.linalg.norm(X_t10 - X)
        diff_20 = np.linalg.norm(X_t20 - X)
        
        # Not strictly monotonic but generally increases
        assert diff_10 > diff_05 * 0.5  # Some deformation occurred
    
    def test_alpha_beta_effect(self):
        """Alpha and beta parameters should control deformation strength."""
        from src.deformation import deform_points
        X = np.array([[1.0, 0.5]])
        
        X_a1 = deform_points(X, t=1.0, alpha=1.0, beta=0.0)
        X_b1 = deform_points(X, t=1.0, alpha=0.0, beta=1.0)
        
        # With alpha=1, beta=0: only x changes
        # With alpha=0, beta=1: only y changes
        x_change_a = abs(X_a1[0, 0] - X[0, 0])
        y_change_a = abs(X_a1[0, 1] - X[0, 1])
        x_change_b = abs(X_b1[0, 0] - X[0, 0])
        y_change_b = abs(X_b1[0, 1] - X[0, 1])
        
        assert x_change_a > y_change_a * 0.5 or y_change_a < 0.01
        assert y_change_b > x_change_b * 0.5 or x_change_b < 0.01


class TestHamiltonians:
    """Tests for Hamiltonian creation and usage."""
    
    def test_create_default(self):
        """Default Hamiltonian should work."""
        from src.deformation import create_hamiltonian, HamiltonianType
        H = create_hamiltonian(HamiltonianType.ZZ_X)
        assert H is not None
    
    def test_create_heisenberg(self):
        """Heisenberg Hamiltonian should work."""
        from src.deformation import create_hamiltonian, HamiltonianType
        H = create_hamiltonian(HamiltonianType.HEISENBERG)
        assert H is not None
    
    def test_create_ising(self):
        """Ising Hamiltonian should work."""
        from src.deformation import create_hamiltonian, HamiltonianType
        H = create_hamiltonian(HamiltonianType.ISING_TRANSVERSE)
        assert H is not None
    
    def test_create_xxz(self):
        """XXZ Hamiltonian should work."""
        from src.deformation import create_hamiltonian, HamiltonianType
        H = create_hamiltonian(HamiltonianType.XXZ, anisotropy=0.5)
        assert H is not None
    
    def test_custom_hamiltonian_in_deform(self):
        """Custom Hamiltonian should be usable in deform_points."""
        from src.deformation import create_hamiltonian, HamiltonianType, deform_points
        H = create_hamiltonian(HamiltonianType.HEISENBERG)
        X = np.array([[1.0, 0.5]])
        X_def = deform_points(X, t=0.5, H=H)
        assert X_def.shape == X.shape


class TestTopologyPreservation:
    """Tests for topology preservation metric."""
    
    def test_identical_preserves_topology(self):
        """Identical points should have perfect preservation."""
        from src.deformation import estimate_topology_preservation
        X = np.random.randn(50, 2)
        score = estimate_topology_preservation(X, X.copy())
        assert np.isclose(score, 1.0)
    
    def test_random_destroys_topology(self):
        """Random scramble should have low preservation."""
        from src.deformation import estimate_topology_preservation
        X = np.random.randn(50, 2)
        X_random = np.random.randn(50, 2)
        score = estimate_topology_preservation(X, X_random)
        # Random should have low score (but not necessarily zero)
        assert score < 0.8
    
    def test_small_perturbation(self):
        """Small perturbation should preserve most topology."""
        from src.deformation import estimate_topology_preservation
        X = np.random.randn(50, 2)
        X_perturbed = X + 0.01 * np.random.randn(50, 2)
        score = estimate_topology_preservation(X, X_perturbed)
        assert score > 0.7  # Should preserve most neighbors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
