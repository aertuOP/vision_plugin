# -*- coding: utf-8 -*-
"""vision_plugin 的 MCP 适配层 (08-16).

把 VisionPipeline 的能力暴露为标准 MCP 工具 (stdio 传输), 让任何 MCP 客户端
直接接入视觉理解能力: 图片分析 (MiniCPM-V) / 实时画面快照 / 行为事件流.

用法::
    # MCP 客户端配置:
    #   { "command": "E:/ai/venv/Scripts/python.exe",
    #     "args": ["-m", "vision_plugin.mcp_server"] }
    #
    # 手动测试:
    #   E:/ai/venv/Scripts/python.exe -m vision_plugin.mcp_server

环境变量:
    VISION_MODEL_DIR   MiniCPM-V 模型目录 (含 transformers/ 或 GGUF)
    VISION_MODEL_ROOT  mediapipe 模型根目录 (含 models/mediapipe/*.task)
    USERPROFILE        chroma 缓存 (不影响本插件, 保留兼容)

默认模型位置: 未设置环境变量时, 自动探测插件包旁的 models/ 目录
(E:\\abc\\models\\minicpm-v46 与 E:\\abc\\models\\mediapipe), 开箱即用.

注意: 图片需以本地文件路径传入 (MCP 工具参数为路径字符串).
本层只做薄封装, 不修改插件核心实现.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from fastmcp import FastMCP  # noqa: E402

from vision_plugin import VisionPipeline  # noqa: E402
from vision_plugin.semantic import describe_image  # noqa: E402

# 全局单例 (MCP 服务器进程常驻, 一次初始化)
_pipe: VisionPipeline | None = None


def _get_pipe() -> VisionPipeline:
    global _pipe
    if _pipe is None:
        # 08-16 单一事实来源: 显式环境变量优先, 否则交给 VisionPipeline 自动探测插件自带模型
        _pipe = VisionPipeline(
            model_root=os.environ.get("VISION_MODEL_ROOT") or None,
            model_dir=os.environ.get("VISION_MODEL_DIR") or None,
            storage=None,
        )
    return _pipe


mcp = FastMCP(
    "vision-plugin",
    instructions=(
        "视觉理解服务。支持: 分析图片文件(analyze_image, MiniCPM-V 语义描述)、"
        "分析实时画面快照(analyze_snapshot)、查看当前画面状态(snapshot)、"
        "获取行为事件流(get_events)。图片用本地文件路径传入。"
    ),
)


def _image_to_b64(path_str: str) -> tuple[str | None, str]:
    """读本地图片 → base64 (JPEG). 返回 (b64, 错误信息)."""
    try:
        p = Path(path_str)
        if not p.exists():
            return None, f"文件不存在: {path_str}"
        raw = p.read_bytes()
        # 统一转 JPEG (MiniCPM 处理器对格式敏感)
        from PIL import Image  # noqa: PLC0415

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode(), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"读取图片失败: {exc}"


@mcp.tool()
def analyze_image(image_path: str, prompt: str = "") -> str:
    """分析一张本地图片 (MiniCPM-V 语义描述)。

    Args:
        image_path: 本地图片文件绝对路径 (jpg/png/bmp/webp)
        prompt: 自定义分析问题 (默认: 用一两句话描述画面内容)
    Returns:
        中文描述 (模型不可用时返回提示)
    """
    b64, err = _image_to_b64(image_path)
    if err:
        return err
    try:
        text = describe_image(b64, prompt or "用一两句话描述这张图片里发生了什么")
        return text or "(模型未返回描述)"
    except Exception as exc:  # noqa: BLE001
        return f"分析失败: {exc}"


@mcp.tool()
def snapshot() -> str:
    """查看当前实时画面状态 (人是否在场/运动强度/事件快照)。

    Returns:
        实时状态摘要
    """
    try:
        st = _get_pipe().snapshot()
        parts = []
        for k, v in (st or {}).items():
            if isinstance(v, (int, float, bool, str)):
                parts.append(f"{k}={v}")
        return "; ".join(parts) or "(无状态)"
    except Exception as exc:  # noqa: BLE001
        return f"获取快照失败: {exc}"


@mcp.tool()
def get_events(since_ts: float = 0.0, limit: int = 5) -> str:
    """获取最近的行为事件 (person_enter / motion_burst / motion_chaos / motion_settle)。

    Args:
        since_ts: 只取该时间戳之后的事件 (0 = 全部缓冲)
        limit: 最大条数
    Returns:
        事件列表 (类型 + 时间戳 + 数据)
    """
    try:
        events = _get_pipe().pop_events(since_ts=since_ts, limit=limit)
        if not events:
            return "(无事件)"
        out = []
        for e in events:
            ts = e.get("ts", "?")
            et = e.get("type", "?")
            data = e.get("data") or {}
            detail = "; ".join(f"{k}={v}" for k, v in data.items() if isinstance(v, (int, float, str)))
            out.append(f"[{et}] ts={ts} {detail}")
        return "\n".join(out)
    except Exception as exc:  # noqa: BLE001
        return f"获取事件失败: {exc}"


def main() -> None:
    mcp.run()  # stdio 传输


if __name__ == "__main__":
    main()
