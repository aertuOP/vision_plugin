# -*- coding: utf-8 -*-
"""vision-plugin — 标准化视觉插件 (A行为检测 + B语义分析 + 级联管线).

模块:
  realtime     MediaPipe 毫秒级行为/人脸检测 (motion_burst/chaos/settle 事件)
  semantic     MiniCPM-V 语义分析 (describe_image/describe_frame)
  pipeline     VisionPipeline: A→B 级联管线 (行为触发语义 → 记忆回调 → 跨模态锚定)
"""
from vision_plugin.pipeline import VisionPipeline
from vision_plugin.realtime import snapshot, pop_events, process_frame, set_model_root
from vision_plugin.semantic import (
    configure_model_dir,
    describe_frame,
    describe_image,
    is_available,
)

__all__ = [
    "VisionPipeline",
    "process_frame", "snapshot", "pop_events", "set_model_root",
    "configure_model_dir", "describe_frame", "describe_image", "is_available",
]
__version__ = "0.1.0"
