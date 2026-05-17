# %% [markdown]
# 可提交的单文件 ARC-AGI-3 Agent（notebook-style cells）
#
# 设计目标：
# 1) 仅一个 py 文件。
# 2) 保留统一接口：is_done(frames, latest_frame) / choose_action(frames, latest_frame)。
# 3) 支持 ACTION6 这类带坐标参数动作（通过 choose_action_payload）。
# 4) 带快速 smoke 测试 + 轻量离线评估（中文输出）。

# %%
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any
import hashlib
import math
import random


# %%
# -----------------------------
# 配置
# -----------------------------
@dataclass
class AgentConfig:
    seed: int = 2026
    max_steps_per_level: int = 360
    stagnation_window: int = 20
    stagnation_unique_ratio: float = 0.30
    forced_reset_after: int = 120
    min_explore_visits: int = 1

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
    default_action6_xy: tuple[int, int] = (0, 0)


# %%
# -----------------------------
# Frame 解析
# -----------------------------
def _frame_get(frame: Any, *keys: str, default: Any = None) -> Any:
    cur = frame
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def extract_grid(frame: Any) -> list[list[int]]:
    grid = _frame_get(frame, "grid", default=None)
    if grid is None:
        grid = _frame_get(frame, "observation", "grid", default=None)
    if grid is None:
        return [[0]]

    if not isinstance(grid, list) or not grid:
        return [[0]]
    if not isinstance(grid[0], list):
        row = [int(max(0, min(15, x))) for x in grid]
        return [row if row else [0]]

    out: list[list[int]] = []
    for row in grid:
        if isinstance(row, list):
            clean = [int(max(0, min(15, x))) for x in row] if row else [0]
            out.append(clean)
    return out if out else [[0]]


def frame_status(frame: Any) -> str:
    status = _frame_get(frame, "status", default=None)
    if status is None:
        status = _frame_get(frame, "game_state", default=None)
    if status is None:
        status = _frame_get(frame, "state", default="NOT_FINISHED")
    return str(status)


def available_actions(frame: Any, cfg: AgentConfig) -> list[str]:
    actions = _frame_get(frame, "available_actions", default=None)
    if actions is None:
        actions = _frame_get(frame, "action_space", default=None)
    if actions is None:
        return list(cfg.action_names)

    out: list[str] = []
    for a in actions:
        if isinstance(a, str):
            out.append(a)
        elif isinstance(a, dict) and "name" in a:
            out.append(str(a["name"]))
    return out or list(cfg.action_names)


def state_hash_from_grid(grid: list[list[int]]) -> str:
    h = hashlib.blake2b(digest_size=16)
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    h.update(f"{rows}x{cols}|".encode())
    for r in grid:
        h.update(bytes(r))
    return h.hexdigest()


# %%
# -----------------------------
# 记忆与统计
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
    recent_states: deque[str] = field(default_factory=lambda: deque(maxlen=80))
    state_visits: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    sa_stats: defaultdict[tuple[str, str], ActionStats] = field(default_factory=lambda: defaultdict(ActionStats))

    pending_state_hash: str | None = None
    pending_action: str | None = None
    pending_entropy: float | None = None


# %%
# -----------------------------
# Agent 主体
# -----------------------------
class EfficientARCAgent:
    """轻量高效策略：新颖度探索 + 进展代理 + 停滞重置。"""

    def __init__(self, cfg: AgentConfig | None = None):
        self.cfg = cfg or AgentConfig()
        self.mem = AgentMemory()
        self.rng = random.Random(self.cfg.seed)

    def _entropy(self, grid: list[list[int]]) -> float:
        hist = [0] * 16
        total = 0
        for row in grid:
            for v in row:
                hist[v] += 1
                total += 1
        if total == 0:
            return 0.0
        ent = 0.0
        for c in hist:
            if c:
                p = c / total
                ent -= p * math.log2(p)
        return ent

    def _center_xy(self, grid: list[list[int]]) -> tuple[int, int]:
        h = len(grid)
        w = len(grid[0]) if h else 1
        return (w // 2, h // 2)

    def _update_previous_transition(self, current_state_hash: str, current_entropy: float) -> None:
        if self.mem.pending_state_hash is None or self.mem.pending_action is None:
            return

        prev_s = self.mem.pending_state_hash
        prev_a = self.mem.pending_action
        prev_e = self.mem.pending_entropy if self.mem.pending_entropy is not None else current_entropy

        changed = 1.0 if current_state_hash != prev_s else -0.25
        entropy_delta = current_entropy - prev_e
        gain = changed + 0.06 * entropy_delta

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
        uniq_ratio = len(set(window)) / max(1, len(window))
        return uniq_ratio < self.cfg.stagnation_unique_ratio

    def _score_action(self, state_hash: str, action: str) -> float:
        stat = self.mem.sa_stats[(state_hash, action)]
        novelty_bonus = 1.0 if stat.visits <= self.cfg.min_explore_visits else 0.0
        explore_term = 0.35 / math.sqrt(1 + stat.visits)
        return stat.mean_gain + novelty_bonus + explore_term

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        status = frame_status(latest_frame)
        if status in {"WIN", "GAME_OVER", "TERMINATED", "DONE"}:
            return True
        return self.mem.step_count >= self.cfg.max_steps_per_level

    def choose_action(self, frames: list[Any], latest_frame: Any) -> str:
        self.mem.step_count += 1
        grid = extract_grid(latest_frame)
        s_hash = state_hash_from_grid(grid)
        entropy = self._entropy(grid)

        self._update_previous_transition(s_hash, entropy)
        self.mem.state_visits[s_hash] += 1
        self.mem.recent_states.append(s_hash)

        if self._is_stagnating() or (self.mem.step_count % self.cfg.forced_reset_after == 0):
            action = self.cfg.reset_action
        else:
            actions = available_actions(latest_frame, self.cfg)
            allowed = set(self.cfg.action_names) | {self.cfg.reset_action}
            clean_actions = [a for a in actions if a in allowed]
            candidates = [a for a in clean_actions if a != self.cfg.reset_action]
            if not candidates:
                candidates = list(self.cfg.action_names)
            order = list(candidates)
            self.rng.shuffle(order)
            action = max(order, key=lambda a: self._score_action(s_hash, a))

        self.mem.pending_state_hash = s_hash
        self.mem.pending_action = action
        self.mem.pending_entropy = entropy
        return action

    def choose_action_payload(self, frames: list[Any], latest_frame: Any) -> Any:
        """可提交辅助接口：若动作为 ACTION6，则返回带 (x,y) 的 payload。"""
        action = self.choose_action(frames, latest_frame)
        if action == "ACTION6":
            x, y = self._center_xy(extract_grid(latest_frame))
            return {"action": "ACTION6", "x": x, "y": y}
        return action


# %%
# -----------------------------
# Smoke 测试 + 离线评估
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


def _logic_smoke_test() -> None:
    agent = EfficientARCAgent()
    frames: list[dict[str, Any]] = []

    for t in range(30):
        fr = _fake_frame(seed=t)
        frames.append(fr)
        if agent.is_done(frames, fr):
            break

        act = agent.choose_action(frames, fr)
        assert isinstance(act, str), "choose_action 必须返回 str"

        payload = agent.choose_action_payload(frames, fr)
        assert isinstance(payload, (str, dict)), "payload 返回类型错误"
        if isinstance(payload, dict):
            assert payload.get("action") == "ACTION6"
            assert isinstance(payload.get("x"), int) and isinstance(payload.get("y"), int)

    final = _fake_frame(seed=999, status="WIN")
    assert agent.is_done(frames, final), "WIN 状态下应终止"


def quick_offline_evaluate(num_episodes: int = 20, horizon: int = 50) -> None:
    """轻量离线评估：仅用于快速逻辑回归，不代表真实 leaderboard。"""
    rng = random.Random(42)
    done_count = 0
    action_hist = defaultdict(int)

    for ep in range(num_episodes):
        agent = EfficientARCAgent(AgentConfig(seed=1000 + ep))
        frames: list[dict[str, Any]] = []
        terminated = False

        for t in range(horizon):
            # 简化环境：后半程随机给 WIN，模拟“可结束回合”
            status = "WIN" if (t > horizon // 2 and rng.random() < 0.04) else "NOT_FINISHED"
            fr = _fake_frame(seed=ep * 100 + t, status=status)
            frames.append(fr)

            if agent.is_done(frames, fr):
                terminated = True
                break

            a = agent.choose_action(frames, fr)
            action_hist[a] += 1

        if terminated:
            done_count += 1

    print("[评估] ===============================")
    print(f"[评估] 回合数: {num_episodes}")
    print(f"[评估] 终止回合数: {done_count}")
    print(f"[评估] 终止率: {done_count / max(1, num_episodes):.2%}")
    top_actions = sorted(action_hist.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print(f"[评估] Top-5 动作频次: {top_actions}")
    print("[评估] 说明: 该评估只验证逻辑稳定性，不等价真实 ARC 分数。")


if __name__ == "__main__":
    _logic_smoke_test()
    print("[smoke] 通过：核心逻辑与返回类型检查完成。")
    quick_offline_evaluate(num_episodes=20, horizon=50)
