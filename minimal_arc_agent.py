# %% [markdown]
# Cell 1 - 项目说明
# ARC-AGI-3 高分向专项求解器（可持续迭代版）
# 本版目标：一次性把关键可升级片段全部落地，形成可继续冲分的统一框架。

# %%
# Cell 2 - 导入依赖
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any
import hashlib
import math
import random


# %%
# Cell 3 - 配置
# -----------------------------
# 配置
# -----------------------------
@dataclass
class AgentConfig:
    seed: int = 2026
    max_steps_per_level: int = 500

    # 停滞与重置
    stagnation_window: int = 26
    stagnation_unique_ratio: float = 0.34
    forced_reset_after: int = 160
    emergency_reset_repeat_state: int = 8

    # 探索与利用
    min_explore_visits: int = 1
    base_exploration_strength: float = 0.50
    base_loop_penalty: float = 0.90
    revisit_penalty_scale: float = 0.08

    # ACTION6
    action6_max_candidates: int = 16
    action6_ratio_soft_cap: float = 0.45
    action6_soft_penalty: float = 0.35

    # profile 参数
    profile_exploration_boost: float = 0.22
    profile_action6_boost: float = 0.30

    reset_action: str = "RESET"
    action_names: tuple[str, ...] = (
        "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"
    )


# %%
# Cell 4 - 工具函数
# -----------------------------
# 工具函数
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
    if grid is None or not isinstance(grid, list) or not grid:
        return [[0]]
    if not isinstance(grid[0], list):
        return [[int(max(0, min(15, x))) for x in grid]]
    out: list[list[int]] = []
    for row in grid:
        if isinstance(row, list):
            out.append([int(max(0, min(15, x))) for x in (row or [0])])
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
    h.update(f"{len(grid)}x{len(grid[0]) if grid else 0}|".encode())
    for r in grid:
        h.update(bytes(r))
    return h.hexdigest()


def game_id_from_frame(frame: Any) -> str:
    gid = _frame_get(frame, "game_id", default=None)
    if gid is None:
        gid = _frame_get(frame, "metadata", "game_id", default=None)
    if gid is None:
        gid = _frame_get(frame, "info", "game_id", default=None)
    return str(gid) if gid is not None else "unknown_game"


def grid_signature(grid: list[list[int]]) -> tuple[int, int, int, int]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    hist = [0] * 16
    for row in grid:
        for v in row:
            hist[v] += 1
    non_zero_colors = sum(1 for c in hist[1:] if c > 0)
    dominant = max(range(16), key=lambda c: hist[c]) if h and w else 0
    return (h, w, non_zero_colors, dominant)


# %%
# Cell 5 - 数据结构
# -----------------------------
# 数据结构
# -----------------------------
@dataclass
class ActionStats:
    visits: int = 0
    total_gain: float = 0.0

    @property
    def mean_gain(self) -> float:
        return self.total_gain / self.visits if self.visits > 0 else 0.0


@dataclass
class ProfileParams:
    exploration_strength: float
    loop_penalty: float
    prefer_action6: float


@dataclass
class AgentMemory:
    step_count: int = 0
    current_game_id: str = "unknown_game"
    current_profile: tuple[int, int, int, int] | None = None

    recent_states: deque[str] = field(default_factory=lambda: deque(maxlen=128))
    recent_actions: deque[str] = field(default_factory=lambda: deque(maxlen=128))
    state_visits: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    action_counts: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))

    sa_stats: defaultdict[tuple[str, str], ActionStats] = field(default_factory=lambda: defaultdict(ActionStats))
    action6_xy_stats: defaultdict[tuple[str, int, int], ActionStats] = field(default_factory=lambda: defaultdict(ActionStats))

    global_sa_stats: defaultdict[tuple[tuple[int, int, int, int], str], ActionStats] = field(default_factory=lambda: defaultdict(ActionStats))
    global_action6_xy_stats: defaultdict[tuple[tuple[int, int, int, int], int, int], ActionStats] = field(default_factory=lambda: defaultdict(ActionStats))

    pending_state_hash: str | None = None
    pending_action: str | None = None
    pending_entropy: float | None = None
    pending_xy: tuple[int, int] | None = None
    pending_profile: tuple[int, int, int, int] | None = None


# %%
# Cell 6 - 核心 Agent
# -----------------------------
# 核心 Agent
# -----------------------------
class EfficientARCAgent:
    def __init__(self, cfg: AgentConfig | None = None):
        self.cfg = cfg or AgentConfig()
        self.mem = AgentMemory()
        self.rng = random.Random(self.cfg.seed)

    def _entropy(self, grid: list[list[int]]) -> float:
        hist = [0] * 16
        total = 0
        for r in grid:
            for v in r:
                hist[v] += 1
                total += 1
        if total == 0:
            return 0.0
        e = 0.0
        for c in hist:
            if c:
                p = c / total
                e -= p * math.log2(p)
        return e

    def _profile_params(self, profile: tuple[int, int, int, int]) -> ProfileParams:
        _, _, non_zero_colors, dominant = profile
        explore = self.cfg.base_exploration_strength
        loop_pen = self.cfg.base_loop_penalty
        a6_boost = 0.0
        if non_zero_colors >= 6:
            explore += self.cfg.profile_exploration_boost
            a6_boost += 0.12
        if dominant == 0:
            a6_boost += self.cfg.profile_action6_boost
        return ProfileParams(exploration_strength=explore, loop_penalty=loop_pen, prefer_action6=a6_boost)

    def _update_previous_transition(self, current_state_hash: str, current_entropy: float) -> None:
        if self.mem.pending_state_hash is None or self.mem.pending_action is None:
            return
        prev_s = self.mem.pending_state_hash
        prev_a = self.mem.pending_action
        prev_e = self.mem.pending_entropy if self.mem.pending_entropy is not None else current_entropy
        prev_p = self.mem.pending_profile

        changed = 1.0 if current_state_hash != prev_s else -0.35
        gain = changed + 0.10 * (current_entropy - prev_e)

        st = self.mem.sa_stats[(prev_s, prev_a)]
        st.visits += 1
        st.total_gain += gain

        if prev_a == "ACTION6" and self.mem.pending_xy is not None:
            x, y = self.mem.pending_xy
            xy = self.mem.action6_xy_stats[(prev_s, x, y)]
            xy.visits += 1
            xy.total_gain += gain

        if prev_p is not None:
            gst = self.mem.global_sa_stats[(prev_p, prev_a)]
            gst.visits += 1
            gst.total_gain += gain
            if prev_a == "ACTION6" and self.mem.pending_xy is not None:
                x, y = self.mem.pending_xy
                gxy = self.mem.global_action6_xy_stats[(prev_p, x, y)]
                gxy.visits += 1
                gxy.total_gain += gain

        self.mem.pending_state_hash = None
        self.mem.pending_action = None
        self.mem.pending_entropy = None
        self.mem.pending_xy = None
        self.mem.pending_profile = None

    def _is_stagnating(self) -> bool:
        if len(self.mem.recent_states) < self.cfg.stagnation_window:
            return False
        window = list(self.mem.recent_states)[-self.cfg.stagnation_window:]
        uniq_ratio = len(set(window)) / len(window)
        return uniq_ratio < self.cfg.stagnation_unique_ratio

    def _is_emergency_loop(self, s_hash: str) -> bool:
        # 同一状态短窗口重复过多则紧急 reset
        recent = list(self.mem.recent_states)[-self.cfg.stagnation_window:]
        return recent.count(s_hash) >= self.cfg.emergency_reset_repeat_state

    def _action6_candidates(self, grid: list[list[int]]) -> list[tuple[int, int]]:
        h = len(grid)
        w = len(grid[0]) if h else 1
        base = {(w // 2, h // 2), (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
                (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)}

        # 颜色热点采样
        non_zero = []
        for y, row in enumerate(grid):
            for x, v in enumerate(row):
                if v != 0:
                    non_zero.append((x, y))
        self.rng.shuffle(non_zero)
        for xy in non_zero[: self.cfg.action6_max_candidates]:
            base.add(xy)

        # 额外加“变化带”采样（行列中点）
        if h > 2 and w > 2:
            base.update({(w // 3, h // 3), (2 * w // 3, h // 3), (w // 3, 2 * h // 3), (2 * w // 3, 2 * h // 3)})

        out = []
        for x, y in base:
            out.append((min(max(0, x), w - 1), min(max(0, y), h - 1)))
        return out

    def _action6_ratio_penalty(self, action: str) -> float:
        total = max(1, sum(self.mem.action_counts.values()))
        a6_ratio = self.mem.action_counts["ACTION6"] / total
        if action == "ACTION6" and a6_ratio > self.cfg.action6_ratio_soft_cap:
            return self.cfg.action6_soft_penalty * (a6_ratio - self.cfg.action6_ratio_soft_cap)
        return 0.0

    def _score_action(self, s_hash: str, action: str, params: ProfileParams, profile: tuple[int, int, int, int]) -> float:
        local = self.mem.sa_stats[(s_hash, action)]
        global_st = self.mem.global_sa_stats[(profile, action)]

        novelty = 1.0 if local.visits <= self.cfg.min_explore_visits else 0.0
        explore = params.exploration_strength / math.sqrt(1 + local.visits + global_st.visits)
        revisit = self.cfg.revisit_penalty_scale * self.mem.state_visits[s_hash]
        a6_bonus = params.prefer_action6 if action == "ACTION6" else 0.0
        a6_pen = self._action6_ratio_penalty(action)

        return 0.60 * local.mean_gain + 0.40 * global_st.mean_gain + novelty + explore + a6_bonus - revisit - a6_pen

    def _score_action6_xy(self, s_hash: str, x: int, y: int, profile: tuple[int, int, int, int]) -> float:
        local = self.mem.action6_xy_stats[(s_hash, x, y)]
        global_xy = self.mem.global_action6_xy_stats[(profile, x, y)]
        novelty = 0.8 if local.visits <= self.cfg.min_explore_visits else 0.0
        explore = 0.28 / math.sqrt(1 + local.visits + global_xy.visits)
        return 0.58 * local.mean_gain + 0.42 * global_xy.mean_gain + novelty + explore

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        status = frame_status(latest_frame)
        if status in {"WIN", "GAME_OVER", "TERMINATED", "DONE"}:
            return True
        return self.mem.step_count >= self.cfg.max_steps_per_level

    def choose_action(self, frames: list[Any], latest_frame: Any) -> str:
        payload = self.choose_action_payload(frames, latest_frame)
        if isinstance(payload, str):
            return payload
        return str(payload.get("action", "ACTION1"))

    def choose_action_payload(self, frames: list[Any], latest_frame: Any) -> Any:
        self.mem.step_count += 1
        grid = extract_grid(latest_frame)
        s_hash = state_hash_from_grid(grid)
        entropy = self._entropy(grid)
        profile = grid_signature(grid)

        gid = game_id_from_frame(latest_frame)
        if gid != self.mem.current_game_id:
            self.mem.current_game_id = gid
            self.mem.current_profile = profile
            self.mem.recent_states.clear()
            self.mem.recent_actions.clear()

        self._update_previous_transition(s_hash, entropy)
        self.mem.state_visits[s_hash] += 1
        self.mem.recent_states.append(s_hash)

        params = self._profile_params(profile)

        if self._is_emergency_loop(s_hash) or self._is_stagnating() or (self.mem.step_count % self.cfg.forced_reset_after == 0):
            action: Any = self.cfg.reset_action
        else:
            actions = available_actions(latest_frame, self.cfg)
            allowed = set(self.cfg.action_names) | {self.cfg.reset_action}
            candidates = [a for a in actions if a in allowed and a != self.cfg.reset_action]
            if not candidates:
                candidates = list(self.cfg.action_names)
            ordered = list(candidates)
            self.rng.shuffle(ordered)
            best = max(ordered, key=lambda a: self._score_action(s_hash, a, params, profile))

            if best == "ACTION6":
                xy_cands = self._action6_candidates(grid)
                self.rng.shuffle(xy_cands)
                xy = max(xy_cands, key=lambda p: self._score_action6_xy(s_hash, p[0], p[1], profile))
                action = {"action": "ACTION6", "x": xy[0], "y": xy[1]}
            else:
                action = best

        # 更新动作计数与pending
        action_name = action if isinstance(action, str) else str(action.get("action", "ACTION1"))
        self.mem.action_counts[action_name] += 1
        self.mem.recent_actions.append(action_name)

        self.mem.pending_state_hash = s_hash
        self.mem.pending_action = action_name
        self.mem.pending_entropy = entropy
        self.mem.pending_profile = profile
        if isinstance(action, dict) and action_name == "ACTION6":
            self.mem.pending_xy = (int(action["x"]), int(action["y"]))
        else:
            self.mem.pending_xy = None

        return action


# %%
# Cell 7 - 测试与评估
# -----------------------------
# 测试与评估
# -----------------------------
def _fake_frame(seed: int, status: str = "NOT_FINISHED", game_id: str = "g00") -> dict[str, Any]:
    rng = random.Random(seed)
    return {
        "grid": [[rng.randint(0, 15) for _ in range(8)] for _ in range(8)],
        "status": status,
        "game_id": game_id,
        "available_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7", "RESET"],
    }


def _logic_smoke_test() -> None:
    agent = EfficientARCAgent()
    frames: list[dict[str, Any]] = []
    for t in range(36):
        gid = "gA" if t < 18 else "gB"
        fr = _fake_frame(seed=t, game_id=gid)
        frames.append(fr)
        if agent.is_done(frames, fr):
            break
        a = agent.choose_action(frames, fr)
        assert isinstance(a, str)
        payload = agent.choose_action_payload(frames, fr)
        assert isinstance(payload, (str, dict))
        if isinstance(payload, dict):
            assert payload.get("action") == "ACTION6"
            assert isinstance(payload.get("x"), int) and isinstance(payload.get("y"), int)
    final = _fake_frame(seed=999, status="WIN")
    assert agent.is_done(frames, final)


def quick_offline_evaluate(num_episodes: int = 30, horizon: int = 70) -> None:
    rng = random.Random(42)
    done_count = 0
    action_hist: defaultdict[str, int] = defaultdict(int)

    for ep in range(num_episodes):
        agent = EfficientARCAgent(AgentConfig(seed=7000 + ep))
        frames: list[dict[str, Any]] = []
        for t in range(horizon):
            status = "WIN" if (t > horizon // 2 and rng.random() < 0.05) else "NOT_FINISHED"
            fr = _fake_frame(seed=ep * 1111 + t, status=status, game_id=f"g{ep % 12:02d}")
            frames.append(fr)
            if agent.is_done(frames, fr):
                done_count += 1
                break
            payload = agent.choose_action_payload(frames, fr)
            if isinstance(payload, str):
                action_hist[payload] += 1
            else:
                action_hist[str(payload.get("action", "ACTION1"))] += 1

    total_actions = sum(action_hist.values())
    a6_ratio = (action_hist.get("ACTION6", 0) / total_actions) if total_actions else 0.0
    print("[评估] =======================================")
    print(f"[评估] 回合数: {num_episodes}")
    print(f"[评估] 终止回合数: {done_count}")
    print(f"[评估] 终止率: {done_count / max(1, num_episodes):.2%}")
    print(f"[评估] ACTION6 占比: {a6_ratio:.2%}")
    print(f"[评估] Top-7 动作频次: {sorted(action_hist.items(), key=lambda kv: kv[1], reverse=True)[:7]}")
    print("[评估] 注意：仅用于逻辑回归，不等价真实榜单分数。")


# %%
# Cell 8 - 真实环境接入
# -----------------------------
# 真实环境接入
# -----------------------------
def build_action_input(action_payload: Any) -> Any:
    try:
        from arcengine import ActionInput
    except Exception:
        return action_payload

    if isinstance(action_payload, str):
        return ActionInput(action=action_payload)
    if isinstance(action_payload, dict):
        action = str(action_payload.get("action", "ACTION1"))
        x = action_payload.get("x")
        y = action_payload.get("y")
        if x is not None and y is not None:
            return ActionInput(action=action, x=int(x), y=int(y))
        return ActionInput(action=action)
    return ActionInput(action="ACTION1")


def frame_to_dict(frame_obj: Any) -> dict[str, Any]:
    if isinstance(frame_obj, dict):
        return frame_obj
    for name in ("model_dump", "dict"):
        fn = getattr(frame_obj, name, None)
        if callable(fn):
            try:
                data = fn()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    out: dict[str, Any] = {}
    for key in ("grid", "status", "available_actions", "game_id", "observation", "state", "game_state"):
        if hasattr(frame_obj, key):
            out[key] = getattr(frame_obj, key)
    return out


def evaluate_on_local_public_environments(environments_dir: str = "environment_files", max_games: int | None = 8, max_steps_per_game: int = 320) -> None:
    try:
        from arc_agi import Arcade, OperationMode
    except Exception as e:
        print(f"[接入] 未安装 arc_agi 依赖，跳过真实环境评估: {e}")
        return

    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=environments_dir)
    env_infos = arcade.get_environments()
    if not env_infos:
        print("[接入] 未发现本地 environments。")
        return

    selected = env_infos[: max_games if max_games is not None else len(env_infos)]
    win_count = 0
    finish_count = 0

    print(f"[接入] 发现环境总数: {len(env_infos)}，本次评估: {len(selected)}")
    for info in selected:
        agent = EfficientARCAgent(AgentConfig(seed=2026))
        frames: list[dict[str, Any]] = []

        wrapper = arcade.make(info.game_id)
        fr = frame_to_dict(wrapper.reset())
        fr["game_id"] = info.game_id
        frames.append(fr)

        status = frame_status(fr)
        for _ in range(max_steps_per_game):
            if agent.is_done(frames, fr):
                break
            payload = agent.choose_action_payload(frames, fr)
            fr = frame_to_dict(wrapper.step(build_action_input(payload)))
            fr["game_id"] = info.game_id
            frames.append(fr)
            status = frame_status(fr)
            if status in {"WIN", "GAME_OVER", "TERMINATED", "DONE"}:
                break

        finish_count += 1
        if status == "WIN":
            win_count += 1
        print(f"[接入] game={info.game_id} 结束状态={status} 步数={len(frames)}")

    print("[接入] =======================================")
    print(f"[接入] 已评估游戏数: {finish_count}")
    print(f"[接入] WIN 数: {win_count}")
    print(f"[接入] WIN 率: {win_count / max(1, finish_count):.2%}")


if __name__ == "__main__":
    _logic_smoke_test()
    print("[smoke] 通过：高分向专项求解器核心逻辑正常。")
    quick_offline_evaluate(num_episodes=30, horizon=70)
