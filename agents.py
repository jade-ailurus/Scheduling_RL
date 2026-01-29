"""
agents.py

Minimal PyTorch implementations:
- DQN (baseline)
- QR-DQN (distributional)

Designed for a SimPy-driven simulator (not Gym). The training code supplies
(s, a, r, s', done) transitions.

Key design choices
------------------
- Small, dependency-light code that can be copied into the repo.
- Deterministic seeding support.
- Optional CVaR action selection for QR-DQN (risk-averse policy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_global_seeds(seed: int) -> None:
    """Best-effort determinism for python/random, numpy, torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on CPU
    # Determinism flags (may reduce performance; ok for experiments)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: float  # 1.0 if terminal else 0.0


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self._buf: Deque[Transition] = deque(maxlen=self.capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: float) -> None:
        self._buf.append(
            Transition(
                s=np.asarray(s, dtype=np.float32),
                a=int(a),
                r=float(r),
                s2=np.asarray(s2, dtype=np.float32),
                done=float(done),
            )
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size > len(self._buf):
            raise ValueError(f"batch_size={batch_size} > buffer_size={len(self._buf)}")
        batch = self._rng.sample(list(self._buf), batch_size)
        s = np.stack([t.s for t in batch], axis=0)
        a = np.asarray([t.a for t in batch], dtype=np.int64)
        r = np.asarray([t.r for t in batch], dtype=np.float32)
        s2 = np.stack([t.s2 for t in batch], axis=0)
        d = np.asarray([t.done for t in batch], dtype=np.float32)
        return s, a, r, s2, d


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Vanilla DQN (no dueling, no PER)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.98,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        target_update_steps: int = 1_000,
        seed: int = 0,
        device: Optional[str] = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_steps = int(target_update_steps)

        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        set_global_seeds(seed)
        self.replay = ReplayBuffer(buffer_size, seed=seed)

        self.q = MLP(self.state_dim, self.action_dim).to(self.device)
        self.q_tgt = MLP(self.state_dim, self.action_dim).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self._learn_steps = 0

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(s)
        return int(torch.argmax(q, dim=1).item())

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: float) -> None:
        self.replay.add(s, a, r, s2, done)

    def learn(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None

        s, a, r, s2, d = self.replay.sample(self.batch_size)

        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(d, dtype=torch.float32, device=self.device)

        q_sa = self.q(s_t).gather(1, a_t).squeeze(1)

        with torch.no_grad():
            q_next = self.q_tgt(s2_t).max(dim=1)[0]
            td_target = r_t + self.gamma * (1.0 - d_t) * q_next

        loss = nn.functional.mse_loss(q_sa, td_target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.opt.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update_steps == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({"q": self.q.state_dict(), "q_tgt": self.q_tgt.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.q_tgt.load_state_dict(ckpt.get("q_tgt", ckpt["q"]))


class QRDQNNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, n_quantiles: int, hidden: int = 128) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.n_quantiles = int(n_quantiles)
        self.backbone = nn.Sequential(
            nn.Linear(self.state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, self.action_dim * self.n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        out = self.head(h)
        return out.view(-1, self.action_dim, self.n_quantiles)


def _quantile_huber_loss(
    pred_quantiles: torch.Tensor,  # [B, N]
    target_quantiles: torch.Tensor,  # [B, N]
    taus: torch.Tensor,  # [N]
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    Quantile regression Huber loss (QR-DQN).

    pred_quantiles: [B, N] for chosen action
    target_quantiles: [B, N] bootstrap target
    """
    # Pairwise TD errors: target_j - pred_i  => [B, N, N]
    td = target_quantiles.unsqueeze(1) - pred_quantiles.unsqueeze(2)

    abs_td = td.abs()
    huber = torch.where(
        abs_td <= kappa,
        0.5 * td.pow(2),
        kappa * (abs_td - 0.5 * kappa),
    )

    # taus: [N] -> [1, N, 1]
    taus = taus.view(1, -1, 1)
    # indicator: 1 if td < 0 (i.e., pred > target), else 0
    indicator = (td.detach() < 0.0).float()
    loss = (torch.abs(taus - indicator) * huber).mean()
    return loss


class QRDQNAgent:
    """Quantile Regression DQN with optional CVaR action selection."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_quantiles: int = 51,
        lr: float = 1e-4,
        gamma: float = 0.98,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        target_update_steps: int = 1_000,
        seed: int = 0,
        device: Optional[str] = None,
        kappa: float = 1.0,
        double_q: bool = True,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.n_quantiles = int(n_quantiles)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_steps = int(target_update_steps)
        self.kappa = float(kappa)
        self.double_q = bool(double_q)

        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        set_global_seeds(seed)
        self.replay = ReplayBuffer(buffer_size, seed=seed)

        self.net = QRDQNNet(self.state_dim, self.action_dim, self.n_quantiles).to(self.device)
        self.tgt = QRDQNNet(self.state_dim, self.action_dim, self.n_quantiles).to(self.device)
        self.tgt.load_state_dict(self.net.state_dict())

        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self._learn_steps = 0

        # Fixed quantile fractions (midpoints)
        # taus: [(0.5/N), (1.5/N), ..., ((N-0.5)/N)]
        taus = (torch.arange(self.n_quantiles, dtype=torch.float32) + 0.5) / float(self.n_quantiles)
        self.taus = taus.to(self.device)

    @torch.no_grad()
    def _q_stats(self, quantiles: torch.Tensor, mode: str, cvar_alpha: float) -> torch.Tensor:
        """
        quantiles: [B, A, N]
        returns: [B, A] aggregated value
        """
        if mode == "mean":
            return quantiles.mean(dim=-1)
        if mode == "cvar":
            # CVaR of *returns* for risk-averse action selection:
            # average of the lowest alpha fraction (worst-case tail).
            k = max(1, int(math.ceil(cvar_alpha * self.n_quantiles)))
            return quantiles[..., :k].mean(dim=-1)
        raise ValueError(f"Unknown risk mode: {mode}")

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        risk_mode: str = "mean",
        cvar_alpha: float = 0.2,
    ) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        quantiles = self.net(s)  # [1, A, N]
        q = self._q_stats(quantiles, mode=risk_mode, cvar_alpha=cvar_alpha)  # [1, A]
        return int(torch.argmax(q, dim=1).item())

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: float) -> None:
        self.replay.add(s, a, r, s2, done)

    def learn(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None

        s, a, r, s2, d = self.replay.sample(self.batch_size)

        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(a, dtype=torch.int64, device=self.device)  # [B]
        r_t = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)  # [B,1]
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)  # [B,1]

        # Current quantiles for taken actions
        q_all = self.net(s_t)  # [B, A, N]
        # Gather: [B, N]
        q_a = q_all[torch.arange(q_all.size(0), device=self.device), a_t, :]

        with torch.no_grad():
            # Next quantiles from target net: [B, A, N]
            q2_tgt_all = self.tgt(s2_t)

            if self.double_q:
                # Online net chooses next action by expected value
                q2_online_all = self.net(s2_t)
                next_act = q2_online_all.mean(dim=-1).argmax(dim=1)  # [B]
            else:
                next_act = q2_tgt_all.mean(dim=-1).argmax(dim=1)  # [B]

            q2 = q2_tgt_all[torch.arange(q2_tgt_all.size(0), device=self.device), next_act, :]  # [B,N]

            # Bellman target distribution
            target = r_t + self.gamma * (1.0 - d_t) * q2  # [B,N] broadcast with r_t
            # Ensure shape: [B, N]
            target = target

        loss = _quantile_huber_loss(q_a, target, taus=self.taus, kappa=self.kappa)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        self.opt.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update_steps == 0:
            self.tgt.load_state_dict(self.net.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({"net": self.net.state_dict(), "tgt": self.tgt.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.tgt.load_state_dict(ckpt.get("tgt", ckpt["net"]))
