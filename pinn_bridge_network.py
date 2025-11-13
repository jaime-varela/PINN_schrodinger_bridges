"""
PINN solver for the coupled forward/backward heat system

    φ_t = -ε Δφ
   ĥφ_t =  +ε Δĥφ

with Schrödinger-style boundary coupling at times t=0 and t=1:

    φ(x,0) ĥφ(x,0) = ρ0(x),   φ(x,1) ĥφ(x,1) = ρ1(x)

Notes
-----
* No derivatives of ρ0, ρ1 are required anywhere. We only evaluate their values
  at sampled spatial points; autograd is used solely for φ, ĥφ.
* The pair (φ, ĥφ) is scale-invariant under (c·φ, ĥφ/c). We softly fix the gauge
  by anchoring the mean of log φ at t=1 to zero (configurable).
* Spatial domain: R^d in theory; in practice we train on a large box Ω = [L-, L+]^d
  sampled uniformly. Optionally add a soft boundary regularizer to discourage
  growth near |x| ≈ L+.

Usage (example)
---------------
Define callable density functions rho0(x: Tensor[B,d]) -> Tensor[B,1] and
rho1(x: Tensor[B,d]) -> Tensor[B,1] that return nonnegative values (not necessarily
normalized). Then run:

    python pinn_biheat_solver.py \
        --dim 2 --eps 0.01 --epochs 20000 --lr 1e-3 \
        --L 4.0 --batch 4096 --width 128 --depth 5

Inside main(), we include a demo with two Gaussian bumps if no user functions
are provided. Replace DemoRho with your own callables as needed.
"""
from __future__ import annotations

import math
import argparse
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Callable, Optional, Tuple, Literal


Tensor = torch.Tensor

# -------------------------
# Networks & utilities
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 128, depth: int = 5, act: str = "tanh"):
        super().__init__()
        acts = {
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "relu": nn.ReLU(inplace=True),
        }
        self.act = acts[act]
        layers = []
        last = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(last, width), self.act]
            last = width
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def time_stack(x: Tensor, t: Tensor) -> Tensor:
    """Concatenate spatial x (B,d) with time t (B,1) -> (B,d+1)."""
    return torch.cat([x, t], dim=-1)


def grad(outputs: Tensor, inputs: Tensor) -> Tensor:
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]


def laplacian(u: Tensor, x: Tensor) -> Tensor:
    """Diagonal Laplacian via double reverse-mode: sum_i d^2 u / dx_i^2.
    u: (B,1), x: (B,d) with requires_grad=True.
    Returns (B,1).
    """
    B, d = x.shape
    grads = grad(u, x)  # (B,d)
    lap = []
    for i in range(d):
        gi = grads[..., i:i+1]
        # dgi/dx_i
        hi = torch.autograd.grad(gi, x, grad_outputs=torch.ones_like(gi),
                                 create_graph=True, retain_graph=True)[0][..., i:i+1]
        lap.append(hi)
    return torch.sum(torch.stack(lap, dim=-1), dim=-1)  # (B,1)


# -------------------------
# Sampling 
# -------------------------
@dataclass
class Domain:
    dim: int
    L: float = 5.0  # half-width of box in each dimension

    def sample_x(self, B: int, device: torch.device) -> Tensor:
        return (2 * torch.rand(B, self.dim, device=device) - 1.0) * self.L

    def boundary_mask(self, x: Tensor, tol: float = 0.9) -> Tensor:
        """Boolean mask for points near the box boundary; used for soft decay reg."""
        return (x.abs() > tol * self.L).any(dim=-1, keepdim=True)


# -------------------------
# PINN model wrapper
# -------------------------
# --- In BiHeatPINN ---------------------------------------------
class BiHeatPINN(nn.Module):
    def __init__(self, dim: int, eps: float, width: int = 128, depth: int = 5, act: str = "tanh"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        in_dim = dim + 1
        self._phi_raw  = MLP(in_dim, 1, width, depth, act)
        self._hphi_raw = MLP(in_dim, 1, width, depth, act)
        self._eps_floor = 1e-6  # numeric floor to avoid log(0)

    def _positivize(self, u: Tensor) -> Tensor:
        # Softplus is smoother than exp and avoids blow-ups; add tiny floor.
        return F.softplus(u, beta=1.0) + self._eps_floor

    def forward(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        z = time_stack(x, t)
        phi  = self._positivize(self._phi_raw(z))
        hphi = self._positivize(self._hphi_raw(z))
        return phi, hphi

    def pde_residuals(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        phi, hphi = self.forward(x, t)
        phi_t  = grad(phi, t)
        hphi_t = grad(hphi, t)
        lap_phi  = laplacian(phi, x)
        lap_hphi = laplacian(hphi, x)
        r1 = phi_t + self.eps * lap_phi
        r2 = hphi_t - self.eps * lap_hphi
        return r1, r2


# -------------------------
# Training utilities
# -------------------------
@dataclass
class TrainConfig:
    dim: int = 2
    eps: float = 0.1
    L: float = 5.0
    batch_interior: int = 4096
    batch_bc: int = 4096
    width: int = 128
    depth: int = 5
    act: str = "tanh"
    lr: float = 1e-3
    epochs: int = 20000 # original 20000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Loss weights
    w_pde: float = 1.0
    w_bc: float = 1.0 # original 5.0
    w_gauge: float = 0.05
    w_decay: float = 0.0  # soft penalty near spatial boundary
    seed: int = 0


def train(
    cfg: TrainConfig,
    rho0: Callable[[Tensor], Tensor],
    rho1: Callable[[Tensor], Tensor],
    X_sampler,
    Y_sampler,
    bc_ratio = 0.5,
    interior_ratio = 0.5,
    callback: Optional[Callable[[int, BiHeatPINN, dict], None]] = None,
) -> BiHeatPINN:
    '''
        Trains the Schrodinger potential PINN and returns the model.
    
    '''
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    #(FIXME: This is important when sampling maybe instantiate it outside of the function)
    dom = Domain(cfg.dim, cfg.L)

    model = BiHeatPINN(cfg.dim, cfg.eps, cfg.width, cfg.depth, cfg.act).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)


        # Interior samples for PDE residuals
        X0_data = X_sampler.sample(cfg.batch_interior).to(device)
        X1_data = Y_sampler.sample(cfg.batch_interior).to(device)
        m = int(cfg.batch_interior * interior_ratio)
        x_int = torch.cat([
            X0_data[torch.randint(len(X0_data),(m,))],
            X1_data[torch.randint(len(X1_data),(m,))],
            dom.sample_x(cfg.batch_interior - 2*m, device)
        ], dim=0)
        t_int = torch.rand(cfg.batch_interior, 1, device=device)
        r1, r2 = model.pde_residuals(x_int, t_int)
        loss_pde = (r1.square().mean() + r2.square().mean())

        # Boundary (in time) samples for coupling constraints
        X0_data_bc = X_sampler.sample(cfg.batch_bc).to(device)
        X1_data_bc = Y_sampler.sample(cfg.batch_bc).to(device)
        m = int(cfg.batch_bc * bc_ratio)
        x_bc = torch.cat([X0_data_bc[torch.randint(len(X0_data_bc),(m,))],
                        X1_data_bc[torch.randint(len(X1_data_bc),(m,))],
                        dom.sample_x(cfg.batch_bc - 2*m, device)], dim=0)

        t0 = torch.zeros(cfg.batch_bc, 1, device=device)
        t1 = torch.ones(cfg.batch_bc, 1, device=device)
        phi0, hphi0 = model(x_bc, t0)
        phi1, hphi1 = model(x_bc, t1)
        # Evaluate rho0, rho1 WITHOUT autograd requirement
        with torch.no_grad():
            r0 = rho0(x_bc).to(device)   # >= 0
            r1v = rho1(x_bc).to(device)  # >= 0

        prod0 = (phi0 * hphi0).squeeze()
        prod1 = (phi1 * hphi1).squeeze()

        eps_guard = 1e-8
        log_prod0 = torch.log(prod0 + eps_guard)
        log_prod1 = torch.log(prod1 + eps_guard)
        log_r0    = torch.log(r0 + eps_guard)
        log_r1    = torch.log(r1v + eps_guard)

        loss_bc = F.mse_loss(log_prod0, log_r0) + F.mse_loss(log_prod1, log_r1)
        # Soft gauge: mean(log φ(x,1)) ≈ 0 to remove global scale
        # Guard small values to avoid log underflow
        eps_guard = 1e-8
        # gauge = (torch.log(phi1 + eps_guard).mean())**2
        # loss_gauge = gauge
        # after computing phi0,hphi0, phi1,hphi1 and eps_guard
        gauge1 = (torch.log(phi1 + eps_guard).mean())**2        # existing
        gauge2 = (torch.log(hphi0 + eps_guard).mean())**2        # NEW: tie ĥφ at t=0
        loss_gauge = gauge1 + gauge2

        # Optional spatial decay near boundary of box
        if cfg.w_decay > 0.0:
            mask = dom.boundary_mask(x_int).float()
            # small magnitude and small gradient near boundary
            phi_b, hphi_b = model(x_int, t_int)
            decay = (mask * (phi_b.square() + hphi_b.square())).mean()
        else:
            decay = torch.tensor(0.0, device=device)

        loss = cfg.w_pde * loss_pde + cfg.w_bc * loss_bc + cfg.w_gauge * loss_gauge + cfg.w_decay * decay
        loss.backward()
        opt.step()

        if epoch % 500 == 0 or epoch == 1:
            with torch.no_grad():
                # Monitor relative BC errors
                bc0_rel = ((phi0 * hphi0 - r0).abs().mean() / (r0.abs().mean() + 1e-8)).item()
                bc1_rel = ((phi1 * hphi1 - r1v).abs().mean() / (r1v.abs().mean() + 1e-8)).item()
                stats = {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "loss_pde": float(loss_pde.item()),
                    "loss_bc": float(loss_bc.item()),
                    "loss_gauge": float(loss_gauge.item()),
                    "bc_rel_0": bc0_rel,
                    "bc_rel_1": bc1_rel,
                }
                print(stats)
                if callback is not None:
                    callback(epoch, model, stats)

    return model




Head = Literal["phi", "hat_phi", "mix", "both"]

def _ensure_batch_t(t_scalar: float, batch: int, device, dtype):
    return torch.full((batch,), float(t_scalar), device=device, dtype=dtype)

def _grad_log_twoheads(
    model: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    x: torch.Tensor,
    t_scalar: float,
    *,
    phi_is_log: bool = False,
    hat_is_log: bool = False,
    eps_div_guard: float = 1e-12,
    which: Head = "phi",
    mix_weights: Optional[Tuple[float, float]] = None,  # (w_phi, w_hat)
):
    """
    Compute ∇_x log φ and/or ∇_x log φ̂ for a two-head model: (phi_t, hat_phi_t) = model(x,t).
    Args:
        which: 
          - "phi": return grad log φ
          - "hat_phi": return grad log φ̂
          - "mix": return w_phi * grad log φ + w_hat * grad log φ̂
          - "both": return (grad log φ, grad log φ̂)
    """
    x_autograd = x.detach().requires_grad_(True)
    t = _ensure_batch_t(t_scalar, x.shape[0], x.device, x.dtype).unsqueeze(-1)
    phi_t, hat_phi_t = model(x_autograd, t)  # shapes: [B] or [B,1] per head
    phi_t = phi_t.view(x.shape[0])
    hat_phi_t = hat_phi_t.view(x.shape[0])

    def grad_log(v, is_log):
        if is_log:
            (g,) = torch.autograd.grad(v.sum(), x_autograd, create_graph=False)
            return g
        else:
            (gv,) = torch.autograd.grad(v.sum(), x_autograd, create_graph=False)
            v_safe = v.clamp_min(eps_div_guard)
            return gv / v_safe.unsqueeze(-1)

    g_phi = grad_log(phi_t, phi_is_log)
    g_hat = grad_log(hat_phi_t, hat_is_log)

    if which == "phi":
        return g_phi
    if which == "hat_phi":
        return g_hat
    if which == "mix":
        w_phi, w_hat = (1.0, 1.0) if mix_weights is None else mix_weights
        return w_phi * g_phi + w_hat * g_hat
    if which == "both":
        return g_phi, g_hat
    raise ValueError(f"Unknown 'which' = {which}")

def simulate_sde_to_t1_twoheads(
    model: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    X0: torch.Tensor,
    eps: float,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    n_steps: int = 2000,
    use_head: Head = "phi",
    phi_is_log: bool = False,
    hat_is_log: bool = False,
    mix_weights: Optional[Tuple[float, float]] = None,  # used only if use_head="mix"
    return_trajectory: bool = False,
    seed: Optional[int] = None,
    gain: float = 1.0,
):
    r"""
    Euler–Maruyama for
        dx = 2ε * score(x,t) dt + √(2ε) dW,
    where score(x,t) = ∇ log φ, ∇ log φ̂, or a mix of the two.
    """
    assert n_steps > 0 and t1 > t0
    dt = (t1 - t0) / n_steps
    sqrt_2eps_dt = (2.0 * eps * dt) ** 0.5

    # FIXME: not working
    # g = None
    # if seed is not None:
    #     g = torch.Generator(device=X0.device).manual_seed(seed)

    x = X0.detach().clone()
    traj = [x.clone()] if return_trajectory else None

    for k in range(n_steps):
        t_k = t0 + k * dt
        score = _grad_log_twoheads(
            model, x, t_k,
            phi_is_log=phi_is_log,
            hat_is_log=hat_is_log,
            which=use_head,
            mix_weights=mix_weights,
        )
        noise = torch.randn_like(x) * sqrt_2eps_dt
        with torch.no_grad():
            x = x + (2.0 * gain * eps) * score * dt + noise
        if return_trajectory:
            traj.append(x.clone())

    if return_trajectory:
        return x, torch.stack(traj, dim=0)
    return x


def simulate_pf_ode_twoheads(
    model, X0, eps, *, n_steps=2000, use_head="phi"
):
    dt = 1.0 / n_steps
    x = X0.detach().clone()
    model.eval()
    traj = [x.clone()]
    for k in range(n_steps):
        t_k = k * dt
        score = _grad_log_twoheads(model, x, t_k, which=use_head)
        # deterministic probability-flow ODE: dx/dt = 2ε ∇ log φ
        x = x + (2.0 * eps) * score * dt
        traj.append(x.clone())
    return x, torch.stack(traj, dim=0)


def diagnose_drift_sampler(model, X0, eps, n_steps=2000, use_head="phi"):
    X = X0.clone()
    dt = 1.0 / n_steps
    sqrt_2eps_dt = (2.0 * eps * dt) ** 0.5

    drift_norms = []
    noise_norms = []
    step_norms  = []
    score_norms = []

    for k in range(n_steps):
        t_k = k * dt
        # compute score with autograd on a detached copy
        score = _grad_log_twoheads(model, X, t_k, which=use_head)

        # record norms (mean over batch)
        sn = score.norm(dim=1).mean().item()
        score_norms.append(sn)

        drift = (2.0 * eps) * score * dt
        noise = torch.randn_like(X) * sqrt_2eps_dt

        drift_norms.append(drift.norm(dim=1).mean().item())
        noise_norms.append(noise.norm(dim=1).mean().item())

        X = X + drift + noise
        step_norms.append((drift + noise).norm(dim=1).mean().item())

    return {
        "score_norm_mean": float(torch.tensor(score_norms).mean()),
        "score_norm_max":  float(torch.tensor(score_norms).max()),
        "drift_norm_mean": float(torch.tensor(drift_norms).mean()),
        "noise_norm_mean": float(torch.tensor(noise_norms).mean()),
        "step_norm_mean":  float(torch.tensor(step_norms).mean()),
    }

