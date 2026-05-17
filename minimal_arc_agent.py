# %% [markdown]
# Minimal ARC-AGI-3 single-file baseline agent (notebook-style cells)
#
# Design goals:
# 1) one file only
# 2) deterministic + robust baseline
# 3) implements is_done(frames, latest_frame) and choose_action(frames, latest_frame)
# 4) lightweight exploration + loop avoidance + reset fallback

# %%
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Any
import hashlib
import random

import math


# %%
# -----------------------------
# Configuration
# -----------------------------
@dataclass
class AgentConfig:
    seed: int = 2026
    max_steps_per_level: int = 350
    stagnation_window: int = 18
    stagnation_unique_ratio: float = 0.28
    forced_reset_after: int = 110
    min_explore_visits: int = 1

    # action names used by ARC-AGI-3 problem statement
    reset_action: str = "RESET"
    action_names: tuple[str, ...] = (
        "ACTION1",
        "ACTION2",
        "ACTION3",
        "ACTION4",
        "ACTION5",
        "ACTION6",
        "ACTION7",
    )


# %%
# -----------------------------
# Helpers for frame parsing
# -----------------------------
def _frame_get(frame: Any, *keys: str, default: Any = None) -> Any:
    """Safely fetch nested keys from dict-like frame objects."""
    cur = frame
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def extract_grid(frame: Any) -> list[list[int]]:
    """
    Convert frame into a (H, W) uint8 grid.
    Supports common ARC frame layouts.
    """
    grid = _frame_get(frame, "grid", default=None)
    if grid is None:
        # fallback layout seen in some envs: frame['observation']['grid']
        grid = _frame_get(frame, "observation", "grid", default=None)
    if grid is None:
        # conservative fallback: empty 1x1 grid
        return [[0]]

    if not isinstance(grid, list) or not grid:
        return [[0]]
    if not isinstance(grid[0], list):
        return [[int(max(0, min(15, x))) for x in grid]]
    out: list[list[int]] = []
    for row in grid:
        if isinstance(row, list) and row:
            out.append([int(max(0, min(15, x))) for x in row])
    return out if out else [[0]]


def frame_status(frame: Any) -> str:
    """Read game status with common key variants."""
    status = _frame_get(frame, "status", default=None)
    if status is None:
        status = _frame_get(frame, "game_state", default=None)
    if status is None:
        status = _frame_get(frame, "state", default="NOT_FINISHED")
    return str(status)


def available_actions(frame: Any, cfg: AgentConfig) -> list[str]:
    """
    Read available actions from frame if provided.
    Otherwise return full default action space.
    """
    actions = _frame_get(frame, "available_actions", default=None)
    if actions is None:
        actions = _frame_get(frame, "action_space", default=None)

    if actions is None:
        return list(cfg.action_names)

    out = []
    for a in actions:
        if isinstance(a, str):
            out.append(a)
        elif isinstance(a, dict) and "name" in a:
            out.append(str(a["name"]))
    return out or list(cfg.action_names)


def state_hash_from_grid(grid: list[list[int]]) -> str:
    """Deterministic state hash for visit counting and loop detection."""
    h = hashlib.blake2b(digest_size=16)
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    h.update(f"{rows}x{cols}|".encode())
    for r in grid:
        h.update(bytes(r))
    return h.hexdigest()


# %%
# -----------------------------
# Minimal memory structures
# -----------------------------
@dataclass
class ActionStats:
    visits: int = 0
    total_gain: float = 0.0

    @property
    def mean_gain(self) -> float:
        return self.total_gain / self.visits if self.visits > 0 else 0.0


@dataclass
class AgentMemory:
    step_count: int = 0
    recent_states: deque[str] = field(default_factory=lambda: deque(maxlen=64))
    state_visits: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    sa_stats: defaultdict[tuple[str, str], ActionStats] = field(default_factory=lambda: defaultdict(ActionStats))
    pending_state_hash: str | None = None
    pending_action: str | None = None
    pending_entropy: float | None = None


# %%
# -----------------------------
# Baseline policy
# -----------------------------
class MinimalARCAgent:
    """
    Deterministic baseline with:
    - novelty-driven exploration
    - simple progress proxy (entropy change)
    - loop/stagnation detection and reset fallback
    """

    def __init__(self, cfg: AgentConfig | None = None):
        self.cfg = cfg or AgentConfig()
        self.mem = AgentMemory()
        self.rng = random.Random(self.cfg.seed)

    def _entropy(self, grid: list[list[int]]) -> float:
        hist = [0] * 16
        total = 0
        for row in grid:
            for v in row:
                vv = int(max(0, min(15, v)))
                hist[vv] += 1
                total += 1
        if total == 0:
            return 0.0
        ent = 0.0
        for c in hist:
            if c:
                p = c / total
                ent -= p * math.log2(p)
        return ent

    def _update_previous_transition(self, current_state_hash: str, current_entropy: float) -> None:
        """Update stats for the last chosen action using current frame as feedback."""
        if self.mem.pending_state_hash is None or self.mem.pending_action is None:
            return

        prev_s = self.mem.pending_state_hash
        prev_a = self.mem.pending_action
        prev_e = self.mem.pending_entropy if self.mem.pending_entropy is not None else current_entropy

        # Proxy gain: state changed + entropy movement (tiny signal)
        changed = 1.0 if current_state_hash != prev_s else -0.2
        gain = changed + 0.05 * (current_entropy - prev_e)

        st = self.mem.sa_stats[(prev_s, prev_a)]
        st.visits += 1
        st.total_gain += gain

        self.mem.pending_state_hash = None
        self.mem.pending_action = None
        self.mem.pending_entropy = None

    def _is_stagnating(self) -> bool:
        if len(self.mem.recent_states) < self.cfg.stagnation_window:
            return False
        window = list(self.mem.recent_states)[-self.cfg.stagnation_window :]
        uniq = len(set(window))
        ratio = uniq / max(1, len(window))
        return ratio < self.cfg.stagnation_unique_ratio

    def _score_action(self, state_hash: str, action: str) -> float:
        visits = self.mem.sa_stats[(state_hash, action)].visits
        mean_gain = self.mem.sa_stats[(state_hash, action)].mean_gain

        # Higher score for under-explored actions in this state
        novelty_bonus = 1.0 if visits <= self.cfg.min_explore_visits else 0.0
        # Gentle UCB-like term
        exploration = 0.35 / math.sqrt(1.0 + visits)
        return mean_gain + novelty_bonus + float(exploration)

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        """Stop on terminal frame states or hard step budget."""
        status = frame_status(latest_frame)
        if status in {"WIN", "GAME_OVER", "TERMINATED", "DONE"}:
            return True
        if self.mem.step_count >= self.cfg.max_steps_per_level:
            return True
        return False

    def choose_action(self, frames: list[Any], latest_frame: Any) -> str:
        """Main decision API expected by ARC-AGI-3 style agents."""
        self.mem.step_count += 1

        grid = extract_grid(latest_frame)
        s_hash = state_hash_from_grid(grid)
        entropy = self._entropy(grid)

        self._update_previous_transition(s_hash, entropy)

        self.mem.state_visits[s_hash] += 1
        self.mem.recent_states.append(s_hash)

        # Safety reset if long stagnation or periodic timeout
        if self._is_stagnating() or (self.mem.step_count % self.cfg.forced_reset_after == 0):
            action = self.cfg.reset_action
        else:
            actions = available_actions(latest_frame, self.cfg)
            # Keep only primary action tokens + reset (if exposed)
            clean_actions = [a for a in actions if a in set(self.cfg.action_names) | {self.cfg.reset_action}]
            candidate_actions = [a for a in clean_actions if a != self.cfg.reset_action]
            if not candidate_actions:
                candidate_actions = list(self.cfg.action_names)

            # Deterministic tie-breaking via shuffled stable copy
            ordered = list(candidate_actions)
            self.rng.shuffle(ordered)
            action = max(ordered, key=lambda a: self._score_action(s_hash, a))

        self.mem.pending_state_hash = s_hash
        self.mem.pending_action = action
        self.mem.pending_entropy = entropy
        return action


# %%
# -----------------------------
# Optional local smoke test (doesn't require ARC runtime)
# -----------------------------
def _fake_frame(seed: int, status: str = "NOT_FINISHED") -> dict[str, Any]:
    rng = random.Random(seed)
    return {
        "grid": [[rng.randint(0, 15) for _ in range(8)] for _ in range(8)],
        "status": status,
        "available_actions": [
            "ACTION1",
            "ACTION2",
            "ACTION3",
            "ACTION4",
            "ACTION5",
            "ACTION6",
            "ACTION7",
            "RESET",
        ],
    }


def _smoke_test() -> None:
    agent = MinimalARCAgent()
    frames: list[dict[str, Any]] = []
    for t in range(20):
        fr = _fake_frame(seed=t)
        frames.append(fr)
        if agent.is_done(frames, fr):
            break
        action = agent.choose_action(frames, fr)
        assert isinstance(action, str)
    final = _fake_frame(seed=999, status="WIN")
    assert agent.is_done(frames, final)


if __name__ == "__main__":
    _smoke_test()
    print("minimal_arc_agent.py smoke test passed")
