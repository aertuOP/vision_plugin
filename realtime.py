# -*- coding: utf-8 -*-
"""实时感知层 (P2-6): MediaPipe 轻量检测 — CPU 毫秒级, 不占 GPU 显存.

职责:
  - 每帧检测人脸/人体 → 结构化信号 (有无人在、几个、活动状态)
  - 变化检测: 人脸出现/消失、大幅动作 → 触发事件 (供语义层分析关键帧)
  - 输出是数据 (不是文字), 兰的对话引擎可直接消费

设计: 常驻线程 + 最新帧缓存, API 拉取当前状态 (零延迟读取).
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 全局状态: 最新检测结果 + 事件队列 (线程安全)
_STATE = {
    "faces": 0,          # 当前检测到的人脸数
    "bodies": 0,         # 当前检测到的人体数
    "activity": "none",  # none / still / moving / busy
    "motion_level": 0.0, # 0-1 画面运动幅度
    "last_update": 0.0,  # 时间戳
    "person_present": False,  # 画面里有没有人
}
_EVENTS: list[dict] = []  # 事件队列: {type, ts, data}
_EVENTS_LOCK = threading.Lock()
_LOCK = threading.Lock()

# 检测器实例 (懒加载)
_detector = None
_prev_faces = 0
_prev_motion_avg = 0.0  # 08-09: 大幅动作触发 (A→B 级联: 运动跃升 → 唤醒语义层)
_prev_activity = ""     # 08-09: 运动熵门控 — activity 状态跳变检测 (busy 进出)
_motion_buf: list[float] = []
_BGSUB_WARMUP = 6      # 08-09: bgsub 首帧瞬态保护 — 初始化后前 N 帧 motion 虚高, 不触发 motion_burst
_warmup_count = 0


class _Detector:
    """MediaPipe Tasks API 人脸/姿态检测器 (1.x 新版; 失败降级 cv2)."""

    def __init__(self) -> None:
        self.face = None
        self.pose = None
        # 08-09 修复: 背景差分 (运动检测) 独立于人脸/姿态 — 原先只在 MediaPipe 失败
        # 时创建, MediaPipe 可用时 _motion() 恒返 0.0 → motion_burst 永不触发.
        import cv2  # noqa: PLC0415

        self._bgsub = cv2.createBackgroundSubtractorMOG2()
        try:
            import mediapipe as mp

            self._mp = mp
            from mediapipe.tasks import python as mp_python  # noqa: PLC0415
            from mediapipe.tasks.python import vision as mp_vision  # noqa: PLC0415

            # 人脸检测 (任务 API 需要模型文件; 用内置 bundled 模型)
            base = mp_python.BaseOptions
            self.face = mp_vision.FaceDetector.create_from_options(
                mp_vision.FaceDetectorOptions(
                    base_options=base(model_asset_path=_find_mp_model("face_detector")),
                    min_detection_confidence=0.5,
                )
            )
            try:
                self.pose = mp_vision.PoseLandmarker.create_from_options(
                    mp_vision.PoseLandmarkerOptions(
                        base_options=base(model_asset_path=_find_mp_model("pose_landmarker_lite")),
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                )
            except Exception:  # noqa: BLE001
                self.pose = None
            logger.info("[realtime] MediaPipe Tasks 就绪 (人脸+姿态)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[realtime] MediaPipe 不可用 (%s), 降级 cv2", exc)
            self.face = None
            self.pose = None

    def detect(self, frame: np.ndarray) -> dict:
        """检测一帧 → {faces, bodies, motion}."""
        faces = bodies = 0
        if self.face is not None:
            try:
                mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                        data=frame[:, :, ::-1].copy())  # BGR→RGB
                res = self.face.detect(mp_img)
                if res.detections:
                    faces = len(res.detections)
                if self.pose is not None:
                    pres = self.pose.detect(mp_img)
                    if pres.pose_landmarks:
                        bodies = 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("[realtime] detect err: %s", exc)
        motion = self._motion(frame)
        return {"faces": faces, "bodies": bodies, "motion": motion}

    def _motion(self, frame: np.ndarray) -> float:
        """背景差分 → 0-1 运动幅度."""
        if self._bgsub is not None:
            fg = self._bgsub.apply(frame)
            return float(np.count_nonzero(fg) / (frame.shape[0] * frame.shape[1]))
        return 0.0


# 08-09 插件化: 模型根目录可配置 (原为相对推导 agent_core 根; 插件由使用方指定)
MODEL_ROOT: str | None = None  # 例: "E:/models" → 找 E:/models/mediapipe/face_detector.task


def set_model_root(root: str | None) -> None:
    """设置 mediapipe 模型根目录 (含 models/mediapipe/ 子目录)."""
    global MODEL_ROOT
    MODEL_ROOT = root


def _find_mp_model(name: str) -> str:
    """找 mediapipe 模型文件 (MODEL_ROOT/models/mediapipe/ 或 mediapipe 包内)."""
    import os  # noqa: PLC0415

    mp_dir = os.path.dirname(__import__("mediapipe").__file__)
    candidates = []
    if MODEL_ROOT:
        candidates.append(str(Path(MODEL_ROOT) / "models" / "mediapipe" / f"{name}.task"))
    # 兜底: mediapipe 包内模型
    candidates += [
        os.path.join(mp_dir, "models", f"{name}.task"),
        os.path.join(mp_dir, "modules", f"{name}.task"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"mediapipe 模型 {name}.task 未找到 (尝试: {candidates})")


def _get_detector():
    global _detector
    if _detector is None:
        _detector = _Detector()
    return _detector


def process_frame(frame: np.ndarray) -> dict:
    """处理一帧 (供后台线程循环调用): 更新全局状态 + 触发变化事件.

    08-09 A→B 级联: 运动幅度从低跃升到高 → push motion_burst 事件
    (语义层据此触发行为-对象分析; 剧烈运动 → activity=busy 供降级判断).
    """
    global _prev_faces, _prev_motion_avg
    det = _get_detector().detect(frame)
    faces = det["faces"]
    motion = det["motion"]

    # 运动平滑 (最近 5 帧均值)
    _motion_buf.append(motion)
    if len(_motion_buf) > 5:
        _motion_buf.pop(0)
    motion_avg = sum(_motion_buf) / len(_motion_buf)

    # 活动状态推断 (0.4 以上视为剧烈运动态 — 独立于人脸判断, 无人脸的高速晃动也判定)
    if motion_avg > 0.4:
        activity = "busy"
    elif faces > 0:
        activity = "moving" if motion_avg > 0.05 else "still"
    else:
        activity = "none"

    with _LOCK:
        _STATE["faces"] = faces
        _STATE["bodies"] = det["bodies"]
        _STATE["activity"] = activity
        _STATE["motion_level"] = round(motion_avg, 3)
        _STATE["person_present"] = faces > 0 or det["bodies"] > 0
        _STATE["last_update"] = time.time()

    # 事件: 人脸出现/消失 (供语义层触发关键帧分析)
    if faces > 0 and _prev_faces == 0:
        _push_event("person_enter", {"faces": faces})
    elif faces == 0 and _prev_faces > 0:
        _push_event("person_leave", {"faces": 0})
    _prev_faces = faces

    # 08-09 事件: 大幅动作触发 (行为-对象分析)
    # 用原始 motion (非平滑): 平滑 5 帧均值滞后, 短促动作峰值会被拉低错过触发.
    # 触发条件: 原始 motion 超阈值 + 5帧缓冲内曾平静 (从静到动的爬升; 持续运动不重复触发)
    global _warmup_count, _prev_activity
    _warmup_count += 1
    if (motion > 0.15 and _motion_buf and min(_motion_buf) < 0.10
            and _warmup_count > _BGSUB_WARMUP):
        _push_event("motion_burst", {"motion": round(motion, 3)})
        logger.info("[realtime] 大幅动作触发 (motion %.2f)", motion)

    # 08-09 运动熵门控: busy(剧烈运动) 进入/退出 状态跳变事件
    # - 进入 busy: 记录剧烈运动态 (语义层降级, 不跑全图分析)
    # - 退出 busy: 运动结束 → 回溯补偿 (语义层重分析静止画面, 绑定之前运动)
    if activity == "busy" and _prev_activity != "busy":
        _push_event("motion_chaos", {"motion": round(motion, 3)})
        logger.info("[realtime] 进入剧烈运动态 (motion %.2f)", motion)
    elif activity != "busy" and _prev_activity == "busy":
        _push_event("motion_settle", {"motion": round(motion, 3)})
        logger.info("[realtime] 剧烈运动结束 (motion %.2f), 回溯补偿", motion)
    _prev_activity = activity
    _prev_motion_avg = motion_avg

    return dict(_STATE)


def _push_event(etype: str, data: dict) -> None:
    with _EVENTS_LOCK:
        _EVENTS.append({"type": etype, "ts": time.time(), "data": data})
        if len(_EVENTS) > 50:
            _EVENTS.pop(0)


def snapshot() -> dict:
    """当前状态快照 (对话引擎/API 拉取, 零延迟)."""
    with _LOCK:
        return dict(_STATE)


def pop_events(since_ts: float = 0.0, limit: int = 5) -> list[dict]:
    """取最近事件 (语义层消费: 有新事件才分析关键帧)."""
    with _EVENTS_LOCK:
        out = [e for e in _EVENTS if e["ts"] > since_ts]
        return out[-limit:]


def start_background(frame_provider, interval: float = 0.1) -> threading.Thread:
    """启动后台循环: 定时从 frame_provider 取帧处理 (interval 秒/帧)."""

    def _loop():
        while True:
            try:
                frame = frame_provider()
                if frame is not None:
                    process_frame(frame)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    logger.info("[realtime] 后台检测循环已启动 (%.2fs/帧)", interval)
    return t
