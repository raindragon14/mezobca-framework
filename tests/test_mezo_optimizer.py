"""
Tests for MeZO optimizer.
"""
import torch
import torch.nn as nn
import numpy as np
import pytest

from src.mezo_optimizer import MeZOOptimizer, MeZOTrainer


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.relu = nn.ReLU()
    
    def forward(self, x, target=None):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        if target is not None:
            loss = nn.functional.mse_loss(x, target)
            return type('Output', (), {'loss': loss})()
        
        return x


def test_mezo_optimizer_creation():
    """Test MeZO optimizer creation."""
    model = SimpleModel()
    optimizer = MeZOOptimizer(
        model,
        lr=1e-3,
        perturbation_epsilon=1e-2,
        sampling_type="antithetic",
        normalize_grad=True,
        device="cpu"
    )
    
    assert optimizer.model == model
    assert optimizer.lr == 1e-3
    assert optimizer.epsilon == 1e-2
    assert optimizer.sampling_type == "antithetic"
    assert optimizer.normalize_grad == True
    assert len(optimizer.parameters) == 4  # 2 weights + 2 biases


def test_perturb_parameters():
    """Test parameter perturbation."""
    model = SimpleModel()
    optimizer = MeZOOptimizer(model, device="cpu")
    
    perturbations = optimizer.perturb_parameters(scale=0.1)
    
    assert len(perturbations) == 4  # All parameters
    for param, pert in perturbations.items():
        assert pert.shape == param.shape
        assert pert.std() > 0  # Should have non-zero variance


def test_apply_perturbations():
    """Test applying perturbations."""
    model = SimpleModel()
    optimizer = MeZOOptimizer(model, device="cpu")
    
    # Store original parameters
    original_params = {name: param.data.clone() for name, param in model.named_parameters()}
    
    # Generate and apply perturbations
    perturbations = optimizer.perturb_parameters(scale=0.1)
    optimizer.apply_perturbations(perturbations, add=True)
    
    # Check parameters changed
    for name, param in model.named_parameters():
        if name in [p for p in original_params]:
            assert not torch.allclose(param.data, original_params[name])
    
    # Apply negative perturbations to restore
    optimizer.apply_perturbations(perturbations, add=False)
    
    # Should be close to original (within numerical error)
    for name, param in model.named_parameters():
        if name in [p for p in original_params]:
            assert torch.allclose(param.data, original_params[name], rtol=1e-5, atol=1e-6)


def test_gradient_estimation_antithetic():
    """Test gradient estimation with antithetic sampling."""
    torch.manual_seed(42)
    
    model = SimpleModel()
    optimizer = MeZOOptimizer(
        model,
        perturbation_epsilon=1e-2,
        sampling_type="antithetic",
        device="cpu"
    )
    
    # Create dummy data
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    
    def loss_fn():
        pred = model(x)
        return nn.functional.mse_loss(pred, y)
    
    # Estimate gradient
    gradients, losses = optimizer.estimate_gradient(loss_fn, return_losses=True)
    
    assert len(gradients) == 4  # All parameters
    assert 'loss_plus' in losses
    assert 'loss_minus' in losses
    
    # Gradients should have correct shapes
    for param, grad in gradients.items():
        assert grad.shape == param.shape
    
    # Losses should be different
    assert losses['loss_plus'] != losses['loss_minus']


def test_gradient_estimation_symmetric():
    """Test gradient estimation with symmetric sampling."""
    torch.manual_seed(42)
    
    model = SimpleModel()
    optimizer = MeZOOptimizer(
        model,
        perturbation_epsilon=1e-2,
        sampling_type="symmetric",
        device="cpu"
    )
    
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    
    def loss_fn():
        pred = model(x)
        return nn.functional.mse_loss(pred, y)
    
    gradients, losses = optimizer.estimate_gradient(loss_fn, return_losses=True)
    
    assert len(gradients) == 4
    assert 'loss_original' in losses
    assert 'loss_perturbed' in losses
    
    for param, grad in gradients.items():
        assert grad.shape == param.shape


def test_normalized_gradients():
    """Test gradient normalization."""
    model = SimpleModel()
    
    # With normalization
    optimizer_norm = MeZOOptimizer(
        model,
        normalize_grad=True,
        device="cpu"
    )
    
    # Without normalization
    optimizer_no_norm = MeZOOptimizer(
        SimpleModel(),  # Fresh model
        normalize_grad=False,
        device="cpu"
    )
    
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    
    def loss_fn():
        pred = model(x)
        return nn.functional.mse_loss(pred, y)
    
    gradients_norm, _ = optimizer_norm.estimate_gradient(loss_fn)
    gradients_no_norm, _ = optimizer_no_norm.estimate_gradient(loss_fn)
    
    # Compute norms
    norm_norm = 0.0
    norm_no_norm = 0.0
    
    for grad in gradients_norm.values():
        norm_norm += grad.norm().item() ** 2
    norm_norm = norm_norm ** 0.5
    
    for grad in gradients_no_norm.values():
        norm_no_norm += grad.norm().item() ** 2
    norm_no_norm = norm_no_norm ** 0.5
    
    # Normalized gradients should have unit norm (approximately)
    assert abs(norm_norm - 1.0) < 0.1
    
    # Unnormalized gradients should have different norm
    assert abs(norm_no_norm - 1.0) > 0.1 or norm_no_norm == 0


def test_optimizer_step():
    """Test optimizer step."""
    torch.manual_seed(42)
    
    model = SimpleModel()
    optimizer = MeZOOptimizer(
        model,
        lr=0.1,
        perturbation_epsilon=1e-2,
        device="cpu"
    )
    
    # Store initial parameters
    initial_params = {name: param.data.clone() for name, param in model.named_parameters()}
    
    # Create dummy data
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    
    def loss_fn():
        pred = model(x)
        return nn.functional.mse_loss(pred, y)
    
    # Estimate gradient and take step
    gradients, _ = optimizer.estimate_gradient(loss_fn)
    optimizer.step(gradients)
    
    # Parameters should have changed
    for name, param in model.named_parameters():
        assert not torch.allclose(param.data, initial_params[name])
    
    # Step count should have increased
    assert optimizer.step_count == 1


def test_multiple_gradient_estimates():
    """Test multiple gradient estimates."""
    model = SimpleModel()
    optimizer = MeZOOptimizer(
        model,
        num_grad_estimates=3,
        device="cpu"
    )
    
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    
    def loss_fn():
        pred = model(x)
        return nn.functional.mse_loss(pred, y)
    
    gradients, _ = optimizer.estimate_gradient(loss_fn)
    
    # Gradients should exist
    assert len(gradients) == 4
    for grad in gradients.values():
        assert torch.isfinite(grad).all()


def test_train_step():
    """Test complete training step."""
    model = SimpleModel()
    optimizer = MeZOOptimizer(
        model,
        lr=0.1,
        device="cpu"
    )
    
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    
    def loss_fn():
        pred = model(x)
        return nn.functional.mse_loss(pred, y)
    
    stats = optimizer.train_step(loss_fn)
    
    assert 'loss' in stats
    assert isinstance(stats['loss'], float)
    
    # Loss should be finite
    assert np.isfinite(stats['loss'])


def test_mezo_trainer_creation():
    """Test MeZOTrainer creation."""
    model = SimpleModel()
    
    # Create dummy dataloader
    from torch.utils.data import DataLoader, TensorDataset
    
    dataset = TensorDataset(torch.randn(20, 10), torch.randn(20, 5))
    dataloader = DataLoader(dataset, batch_size=4)
    
    trainer = MeZOTrainer(
        model=model,
        train_dataloader=dataloader,
        optimizer_config={
            'lr': 1e-3,
            'perturbation_epsilon': 1e-2
        },
        device="cpu"
    )
    
    assert trainer.model == model
    assert trainer.train_dataloader == dataloader
    assert isinstance(trainer.optimizer, MeZOOptimizer)


def test_mezo_trainer_train_epoch():
    """Test trainer training epoch."""
    torch.manual_seed(42)
    
    model = SimpleModel()
    
    # Create dummy dataloader
    from torch.utils.data import DataLoader, TensorDataset
    
    dataset = TensorDataset(torch.randn(20, 10), torch.randn(20, 5))
    dataloader = DataLoader(dataset, batch_size=4)
    
    trainer = MeZOTrainer(
        model=model,
        train_dataloader=dataloader,
        device="cpu"
    )
    
    # Train for one epoch
    stats = trainer.train_epoch(epoch=0)
    
    assert 'train_loss' in stats
    assert np.isfinite(stats['train_loss'])
    
    # Global step should have increased
    assert trainer.global_step == len(dataloader)


def test_memory_efficiency_compared_to_adam():
    """Test that MeZO uses less memory than Adam."""
    import torch.cuda as cuda
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    model = SimpleModel().cuda()
    
    # Adam memory usage
    torch.cuda.reset_peak_memory_stats()
    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    x = torch.randn(4, 10).cuda()
    y = torch.randn(4, 5).cuda()
    
    # Forward
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    
    # Backward (memory intensive)
    loss.backward()
    adam.step()
    adam.zero_grad()
    
    adam_memory = cuda.max_memory_allocated()
    
    # MeZO memory usage
    torch.cuda.reset_peak_memory_stats()
    model2 = SimpleModel().cuda()
    mezo = MeZOOptimizer(model2, device="cuda")
    
    def loss_fn():
        pred = model2(x)
        return nn.functional.mse_loss(pred, y)
    
    gradients, _ = mezo.estimate_gradient(loss_fn)
    mezo.step(gradients)
    
    mezo_memory = cuda.max_memory_allocated()
    
    # MeZO should use less memory (no backward pass)
    assert mezo_memory < adam_memory


def test_convergence_on_quadratic():
    """Test convergence on simple quadratic problem."""
    torch.manual_seed(42)
    
    # Simple quadratic: f(w) = 0.5 * ||w - target||^2
    class QuadraticModel(nn.Module):
        def __init__(self, dim=5):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(dim))
            self.target = torch.randn(dim)
        
        def forward(self):
            return 0.5 * torch.sum((self.w - self.target) ** 2)
    
    model = QuadraticModel(dim=10)
    target = model.target.clone()
    
    optimizer = MeZOOptimizer(
        model,
        lr=0.1,
        perturbation_epsilon=0.1,
        sampling_type="antithetic",
        device="cpu"
    )
    
    initial_loss = model().item()
    
    # Train for some steps
    for step in range(100):
        def loss_fn():
            return model()
        
        gradients, _ = optimizer.estimate_gradient(loss_fn)
        optimizer.step(gradients)
    
    final_loss = model().item()
    
    # Loss should decrease
    assert final_loss < initial_loss
    
    # Should get close to optimum
    w_norm = torch.norm(model.w.data - target).item()
    assert w_norm < 0.5  # Should be close to target


if __name__ == "__main__":
    pytest.main([__file__, "-v"])