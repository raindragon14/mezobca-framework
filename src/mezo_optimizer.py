"""
Memory-Efficient Zeroth-Order Optimization (MeZO) for fine-tuning LLMs on CPU.
Based on: "Memory-Efficient Zeroth-Order Optimization for Large Language Models"
(MeZO: https://arxiv.org/abs/2305.14133)

Key idea: Estimate gradients using only forward passes via perturbations.
Saves memory by avoiding backpropagation. Parameters are restored via PRNG seed
replay so no O(N) parameter clone is needed — true O(1) memory footprint.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Callable

from .logger_utils import logger


class MeZOOptimizer:
    """
    True MeZO ZO-SGD optimizer.

    Memory footprint is O(1) above inference because:
    - No gradient tensors are stored
    - No Adam momentum buffers (exp_avg / exp_avg_sq)
    - Perturbation noise z is regenerated on-the-fly from a PRNG seed

    Sampling modes:
    - "antithetic": 2 forward passes (±ε), lower variance, recommended
    - "symmetric":  2 forward passes (0 and +ε), simpler
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        perturbation_epsilon: float = 1e-3,
        sampling_type: str = "antithetic",
        normalize_grad: bool = True,
        num_grad_estimates: int = 1,
        weight_decay: float = 0.0,
        device: str = "cpu",
        # legacy kwargs silently ignored (betas, eps, maximize)
        **kwargs
    ):
        self.model = model
        self.lr = float(lr)
        self.epsilon = float(perturbation_epsilon)
        self.sampling_type = sampling_type
        self.normalize_grad = normalize_grad
        self.num_grad_estimates = num_grad_estimates
        self.weight_decay = float(weight_decay)
        self.device = device
        self.step_count = 0

        # Divergence mitigation (MeZO paper §4.1):
        # Clip projected gradient to prevent exploding step sizes
        self.grad_clip = kwargs.get("grad_clip", 1.0)  # default: clip at ±1.0
        # Warmup steps: scale LR from 0 → lr over first N steps
        self.warmup_steps = int(kwargs.get("warmup_steps", 0))

        self.parameters = list(model.parameters())
        self.param_groups = [{'params': self.parameters}]

    # ------------------------------------------------------------------
    # Core perturbation primitive
    # ------------------------------------------------------------------

    def apply_seeded_perturbations(self, seed: int, scale: float = 1.0, add: bool = True):
        """
        Add or subtract ε·z to every trainable parameter, where z ~ N(0,I)
        is deterministically generated from `seed`.  Calling this twice with
        (add=True) then (add=False) restores original weights exactly —
        no backup tensor required.
        """
        for param in self.parameters:
            if not param.requires_grad:
                continue
            gen = torch.Generator(device=param.device)
            gen.manual_seed(seed)
            z = torch.randn(param.shape, generator=gen,
                            device=param.device, dtype=param.dtype)
            delta = z * (self.epsilon * scale)
            if add:
                param.data.add_(delta)
            else:
                param.data.sub_(delta)

    # ------------------------------------------------------------------
    # Gradient estimation  (2 forward passes only)
    # ------------------------------------------------------------------

    def estimate_gradient(
        self,
        loss_fn: Callable[[], torch.Tensor],
        return_losses: bool = False,
    ) -> Tuple[float, int, Optional[Dict]]:
        """
        Estimate the projected gradient scalar ĝ and the seed used.

        For antithetic:   ĝ = (L(θ+εz) - L(θ-εz)) / 2ε
        For symmetric:    ĝ = (L(θ+εz) - L(θ))    / ε

        Returns (projected_grad, seed, losses_dict or None).
        No gradient tensor of shape [N] is ever materialised.
        """
        self.model.eval()
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
        losses: Dict = {}

        if self.sampling_type == "antithetic":
            self.apply_seeded_perturbations(seed, add=True)
            with torch.no_grad():
                loss_plus = loss_fn().item()

            # two subtractions move from +ε to -ε
            self.apply_seeded_perturbations(seed, add=False)
            self.apply_seeded_perturbations(seed, add=False)
            with torch.no_grad():
                loss_minus = loss_fn().item()

            # restore to θ
            self.apply_seeded_perturbations(seed, add=True)

            projected_grad = (loss_plus - loss_minus) / (2 * self.epsilon)
            if return_losses:
                losses = {"loss_plus": loss_plus, "loss_minus": loss_minus}

        elif self.sampling_type == "symmetric":
            with torch.no_grad():
                loss_orig = loss_fn().item()

            self.apply_seeded_perturbations(seed, add=True)
            with torch.no_grad():
                loss_pert = loss_fn().item()

            self.apply_seeded_perturbations(seed, add=False)

            projected_grad = (loss_pert - loss_orig) / self.epsilon
            if return_losses:
                losses = {"loss_original": loss_orig, "loss_perturbed": loss_pert}
        else:
            raise ValueError(f"Unknown sampling_type: {self.sampling_type!r}")

        # Divergence mitigation: clip projected gradient
        # Paper insight: ZO estimates can be high-variance; clipping stabilises early training
        if self.grad_clip > 0:
            projected_grad = max(-self.grad_clip, min(self.grad_clip, projected_grad))

        return projected_grad, seed, losses if return_losses else None

    # ------------------------------------------------------------------
    # Parameter update  (ZO-SGD)
    # ------------------------------------------------------------------

    def step(self, projected_grad: float, seed: int):
        """
        θ ← θ - lr · ĝ · z,  where z is regenerated from seed.
        Identical to SGD on a random subspace.

        Paper §3 (MeZO): update every parameter with the same scalar ĝ
        projected onto its own iid Gaussian direction z.
        """
        self.step_count += 1

        # LR warmup: linearly ramp up LR for the first warmup_steps
        if self.warmup_steps > 0 and self.step_count <= self.warmup_steps:
            effective_lr = self.lr * (self.step_count / self.warmup_steps)
        else:
            effective_lr = self.lr
        for param in self.parameters:
            if not param.requires_grad:
                continue
            gen = torch.Generator(device=param.device)
            gen.manual_seed(seed)
            z = torch.randn(param.shape, generator=gen,
                            device=param.device, dtype=param.dtype)
            if self.weight_decay != 0:
                param.data.mul_(1.0 - effective_lr * self.weight_decay)
            param.data.sub_(z, alpha=effective_lr * projected_grad)

    # ------------------------------------------------------------------
    # Convenience: one complete training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        loss_fn: Callable[[], torch.Tensor],
        closure: Optional[Callable] = None,   # kept for API compatibility
    ) -> Dict[str, float]:
        """
        Estimate gradient + update parameters in one call.
        Loss re-used from gradient estimation — no 3rd forward pass.
        """
        projected_grad, seed, losses = self.estimate_gradient(
            loss_fn, return_losses=True
        )
        self.step(projected_grad, seed)

        # Primary loss: first available measurement (no extra forward pass)
        primary = losses.get("loss_plus", losses.get("loss_original", 0.0))
        return {"loss": primary, **losses}