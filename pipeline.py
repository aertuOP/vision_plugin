# -*- coding: utf-8 -*-
"""VisionPipeline — A(行为检测)→B(语义分析) 级联视觉管线 (插件核心).

从兰项目 router.py /vision/feed 的级联逻辑抽离改造 — 零 agent_core 依赖.

事件模型 (由 realtime 检测产生):
  person_enter / person_leave  人脸进出
  motion_burst                 大幅动作 (从静到动) → 触发行为-对象语义分析
  motion_chaos                 进入剧烈运动 (motion>0.4) → 降级记录, 不跑全图
  motion_settle                剧烈运动结束 → 回溯补偿 (重分析静止画面)

用法::
    from vision_plugin import VisionPipeline
    # 方式1: 不传模型参数 → 自动探测插件自带 models/ 目录 (开箱即用)
    pipe = VisionPipeline(storage=mem)
    # 方式2: 显式指定模型目录 (自定义路径时)
    pipe = VisionPipeline(model_root="<根目录>", model_dir="<minicpm目录>", storage=mem)
    st = pipe.feed(frame)                       # 每帧喂入 → 检测 + 事件级联
    desc = pipe.analyze(frame, "描述这张图")     # 主动语义分析
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# 语义触发冷却 (秒) — B 分析 3s+, 事件触发后冷却防刷
DEFAULT_SEM_COOLDOWN = 10.0
# chaos/settle 记录冷却 — 同一事件被多次 feed 轮询会重复处理
DEFAULT_CHAOS_COOLDOWN = 30.0
# 视觉对象↔记忆关联冷却 (秒) — 防打扰
DEFAULT_ANCHOR_COOLDOWN = 1800.0

# 锚定黑名单: 无信息量泛词不触发视觉对象→记忆唤醒
ANCHOR_STOP_WORDS = {"背景", "环境", "画面", "东西", "物体", "手", "桌子", "地面", "空气", "人"}


class VisionPipeline:
    """A→B 级联视觉管线. storage 可选 (None 时只检测不记忆/不锚定)."""

    # 插件自带模型目录 (与插件包同级: <插件根>/models) — 08-16 支持开箱即用
    _PLUGIN_MODELS = Path(__file__).resolve().parent.parent / "models"

    def __init__(
        self,
        model_root: str | Path | None = None,
        model_dir: str | Path | None = None,
        storage=None,
        sem_cooldown: float = DEFAULT_SEM_COOLDOWN,
        chaos_cooldown: float = DEFAULT_CHAOS_COOLDOWN,
        anchor_cooldown: float = DEFAULT_ANCHOR_COOLDOWN,
    ) -> None:
        """初始化.

        Args:
            model_root: mediapipe 模型根目录 (含 models/mediapipe/*.task).
                        不传时自动探测插件自带 models/ 目录 (存在才启用; 无模型优雅降级 cv2)
            model_dir: MiniCPM-V 模型目录 (含 transformers/ 或 GGUF).
                       不传时自动探测插件自带 models/minicpm-v46 (存在才启用; 无模型语义返回 None)
            storage: 记忆对象 (鸭子类型: remember(text, room, metadata)/recall 可选)
            sem_cooldown: 语义分析冷却 (秒)
            chaos_cooldown: 剧烈运动记录冷却 (秒)
            anchor_cooldown: 视觉对象↔记忆关联冷却 (秒)
        """
        import numpy as np  # noqa: PLC0415

        self._np = np
        from . import realtime, semantic  # noqa: PLC0415

        # 08-16 默认模型探测: 显式传入优先; 否则探测插件自带 models/ (开箱即用, 跨机器无需改路径)
        if model_root:
            realtime.set_model_root(str(model_root))
        elif self._PLUGIN_MODELS.exists():
            realtime.set_model_root(str(self._PLUGIN_MODELS.parent))  # realtime 拼 <root>/models/mediapipe/
        if model_dir:
            semantic.configure_model_dir(str(model_dir))
        elif self._PLUGIN_MODELS.exists():
            semantic.configure_model_dir(str(self._PLUGIN_MODELS / "minicpm-v46"))
        self._realtime = realtime
        self._semantic = semantic
        self.storage = storage
        self._sem_cooldown = sem_cooldown
        self._chaos_cooldown = chaos_cooldown
        self._anchor_cooldown = anchor_cooldown
        self._last_sem_ts = 0.0
        self._last_chaos_ts = 0.0
        self._last_anchor_ts = 0.0
        self._warned_unavailable = False

    # ── 主入口: 每帧喂入 ──
    def feed(self, frame) -> dict:
        """处理一帧 (BGR ndarray) → 实时状态 + 事件级联 (返回 {ok, semantic_triggered, ...})."""
        st = self._realtime.process_frame(frame)
        events = self._realtime.pop_events(since_ts=time.time() - 60, limit=4)
        now = time.time()
        triggered = False
        motion_lv = float(st.get("motion_level") or 0.0)

        # ① 运动幅度门控: 进入剧烈运动 → 降级记录 (不分析具体对象)
        if (any(e["type"] == "motion_chaos" for e in events)
                and (now - self._last_chaos_ts) >= self._chaos_cooldown):
            try:
                evt = next(e for e in events if e["type"] == "motion_chaos")
                self._store(
                    f"[运动事件·{self._iso(evt['ts'])}] 画面剧烈运动，无法分辨具体对象"
                    f"（运动幅度 {evt.get('data', {}).get('motion', '?')}）",
                    person=bool(st.get("person_present")),
                )
                self._last_chaos_ts = now
                logger.info("[pipeline] motion_chaos: 剧烈运动降级记录")
            except Exception:  # noqa: BLE001
                pass

        # ② 语义分析触发 (person_enter | motion_burst | motion_settle, 冷却 + 非剧烈)
        if (any(e["type"] in ("person_enter", "motion_burst", "motion_settle") for e in events)
                and (now - self._last_sem_ts) >= self._sem_cooldown
                and motion_lv <= 0.4):
            try:
                evt = next((e for e in events
                            if e["type"] in ("person_enter", "motion_burst", "motion_settle")), None)
                evt_ts = evt.get("ts") if evt else now
                evt_type = evt.get("type", "") if evt else ""
                t0 = self._iso(evt_ts)
                person = bool(st.get("person_present"))

                if evt_type == "motion_burst":
                    # 行为-对象元组 (prompt 引导 VLM)
                    desc = self.analyze(
                        frame,
                        "画面里有什么物体正在被移动/抓取/抛掷？用\"动作:xx 对象:xx\"格式回答，"
                        "如\"动作:抓取 对象:苹果\"。看不清就只给动作。",
                    )
                    if desc:
                        self._store(f"[行为事件·{t0}] {desc}", person=person)
                        triggered = True
                        logger.info("[pipeline] motion_burst→B: %.70s", desc)
                        self._anchor_retrieve(desc)
                elif evt_type == "motion_settle":
                    desc = self.analyze(frame, "用一两句话描述画面里的人在做什么、周围环境")
                    if desc:
                        self._store(f"[运动结束·{t0}] 画面恢复，{desc}", person=person)
                        triggered = True
                        logger.info("[pipeline] motion_settle→B 回溯补偿: %.60s", desc)
                else:  # person_enter
                    desc = self.analyze(frame)
                    if desc:
                        self._store(desc, person=True)
                        triggered = True
                        logger.info("[pipeline] person_enter→B 场景: %.60s", desc)
                if triggered:
                    self._last_sem_ts = now
            except Exception:  # noqa: BLE001
                pass
        return {"ok": True, "realtime": st, "semantic": triggered, "motion_level": motion_lv}

    # ── 主动分析 ──
    def analyze(self, frame, prompt: str = "") -> str | None:
        """MiniCPM-V 语义分析 (帧 → 描述)."""
        if not self._semantic.is_available():
            if not self._warned_unavailable:
                logger.warning("[pipeline] MiniCPM 模型不可用 (用 configure_model_dir 指定模型目录)")
                self._warned_unavailable = True
            return None
        return self._semantic.describe_frame(frame, prompt)

    def snapshot(self) -> dict:
        """当前实时状态 (对话引擎/API 拉取)."""
        return self._realtime.snapshot()

    def pop_events(self, since_ts: float = 0.0, limit: int = 5) -> list[dict]:
        return self._realtime.pop_events(since_ts, limit)

    # ── 内部 ──
    def _store(self, text: str, person: bool = False) -> None:
        """存视觉记忆 (storage 鸭子类型: remember(text, room, metadata))."""
        if self.storage is None:
            return
        import datetime as _dt  # noqa: PLC0415
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        self.storage.remember(
            f"[视觉记忆·{ts}] {text}",
            room="vision",
            metadata={"source": "vision", "ts": ts, "person": "1" if person else "0"},
            dedupe=False,
        )

    def _anchor_retrieve(self, desc: str) -> None:
        """视觉对象↔记忆关联: 对象 → 检索历史 → 强关联命中 → 存锚定记忆 (供 proactive 主动开口)."""
        if self.storage is None or not hasattr(self.storage, "recall_with_scores"):
            return
        import re  # noqa: PLC0415
        now = time.time()
        if now - self._last_anchor_ts < self._anchor_cooldown:
            return
        m = re.search(r"对象[:：]\s*(\S{1,12})", desc)
        if not m:
            return
        obj = m.group(1).strip("，。；!?！？,.;: ")
        if not obj or obj in ANCHOR_STOP_WORDS:
            return
        try:
            hits = self.storage.recall_with_scores(obj, n_results=5, room=None)
            best = None
            for r in hits:
                if r.get("room") == "vision":
                    continue  # 排除刚存的视觉记忆
                sim = float(r.get("similarity") or 0.0)
                if best is None or sim > best[0]:
                    best = (sim, str(r.get("text", ""))[:60])
            if not best:
                return
            sim, hit_text = best
            # 双通道: 字面共词 或 向量高分 (MiniLM 中文短词区分度差, 宁可漏报不误报)
            if not (obj in hit_text or sim >= 0.60):
                return
            self._last_anchor_ts = now
            import datetime as _dt  # noqa: PLC0415
            ts = _dt.datetime.now().isoformat(timespec="seconds")
            self.storage.remember(
                f"[锚定·{ts}] 看到「{obj}」，联想到「{hit_text}」",
                room="vision",
                metadata={"source": "anchor", "anchor_obj": obj, "ts": ts},
                dedupe=False,
            )
            logger.info("[pipeline] 视觉对象↔记忆关联: 「%s」↔「%.40s」(sim=%.2f)", obj, hit_text, sim)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[pipeline] 锚定检索失败: %s", exc)

    @staticmethod
    def _iso(ts: float) -> str:
        import datetime as _dt  # noqa: PLC0415
        return _dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


__all__ = ["VisionPipeline"]
