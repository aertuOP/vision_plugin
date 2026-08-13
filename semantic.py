# -*- coding: utf-8 -*-
"""语义层 (P2-6): MiniCPM-V 4.6 关键帧描述 — CPU 推理, 事件触发.

职责:
  - 接收一帧图像 → 用 MiniCPM-V 描述画面内容 (中文自然语言)
  - 由实时层事件 (person_enter/画面变化) 或用户主动请求触发
  - 输出描述进对话 + 视觉记忆层

实现: llama-cpp-python (GGUF 多模态) 或 transformers (safetensors).
优先 llama.cpp (CPU 2GB 内存, Q4_K_M).
"""
from __future__ import annotations

import base64
import logging
import re
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 08-09 插件化: 模型目录可配置 (原为相对推导 agent_core 根)
MODEL_DIR: Path = Path("models/minicpm-v46")  # 使用方用 configure_model_dir() 指定
MODEL_PATH = MODEL_DIR / "MiniCPM-V-4_6-Q4_K_M.gguf"
MMPROJ_PATH = MODEL_DIR / "mmproj-model-f16.gguf"
TF_DIR = MODEL_DIR / "transformers"  # transformers safetensors 版 (优先)

# 推理锁 (CPU 单实例, 防并发)
_INFER_LOCK = threading.Lock()
_llm = None  # 懒加载
_loaded_at = 0.0


def configure_model_dir(model_dir: str | Path) -> None:
    """设置 MiniCPM-V 模型目录 (含 MiniCPM-V-4_6-Q4_K_M.gguf + transformers/)."""
    global MODEL_DIR, MODEL_PATH, MMPROJ_PATH, TF_DIR, _llm
    _llm = None  # 强制重载
    MODEL_DIR = Path(model_dir)
    MODEL_PATH = MODEL_DIR / "MiniCPM-V-4_6-Q4_K_M.gguf"
    MMPROJ_PATH = MODEL_DIR / "mmproj-model-f16.gguf"
    TF_DIR = MODEL_DIR / "transformers"


def is_available() -> bool:
    return (TF_DIR / "config.json").exists() or (MODEL_PATH.exists() and MMPROJ_PATH.exists())


def _load():
    """懒加载 MiniCPM-V 4.6 (transformers 优先, llama.cpp 备用)."""
    global _llm, _loaded_at
    if _llm is not None:
        return _llm
    if not is_available():
        logger.warning("[semantic] MiniCPM 模型未下载 (%s)", MODEL_DIR)
        return None
    # ① transformers (CPU, 官方支持, Windows 无编译问题)
    _llm = _load_transformers()
    if _llm is not None:
        return _llm
    # ② llama.cpp 备用 (需要 llama-cpp-python 已编译)
    try:
        from llama_cpp import Llama  # noqa: PLC0415

        _llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=0,
            chat_format="minicpm_v",
            mmproj=str(MMPROJ_PATH),
            verbose=False,
        )
        _loaded_at = time.time()
        logger.info("[semantic] MiniCPM-V 4.6 已加载 (llama.cpp CPU)")
        return _llm
    except Exception as exc:  # noqa: BLE001
        logger.warning("[semantic] llama.cpp 不可用: %s", exc)
        return None


def _load_transformers():
    """transformers 加载.

    实测 (AMD 8745H CPU): BF16 CPU 2.8s/张 最快; NF4 反量化开销反而慢 (4.8s).
    策略: BF16 主路径; 08-07 增强: GPU 有空余 (CUDA 可用 + 显存 ≥ 3GB) 时 BF16 进 GPU
    (推理快 3-5 倍), 否则保持 CPU — 自动检测, 不手动指定.
    """
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoProcessor  # noqa: PLC0415
        from transformers.models.minicpmv4_6 import MiniCPMV4_6ForConditionalGeneration  # noqa: PLC0415

        if not (TF_DIR / "config.json").exists():
            logger.warning("[semantic] transformers 模型未下载")
            return None
        # 08-07: 自动检测 GPU 空余 (显存 ≥ 3GB 才用, 避免 OOM 后回退抖动)
        device, dtype = "cpu", torch.bfloat16
        if torch.cuda.is_available():
            try:
                free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
                if free_mb >= 3072:
                    device, dtype = "cuda:0", torch.bfloat16
                    logger.info("[semantic] GPU 显存充足 (%dMB), BF16 进 GPU", free_mb)
                else:
                    logger.info("[semantic] GPU 显存不足 (%dMB < 3072MB), 保持 CPU", free_mb)
            except Exception:  # noqa: BLE001
                pass
        logger.info("[semantic] 加载 MiniCPM-V 4.6 (BF16 %s)...", device)
        model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
            str(TF_DIR),
            dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        proc = AutoProcessor.from_pretrained(str(TF_DIR), trust_remote_code=True)
        _loaded_at = time.time()
        logger.info("[semantic] MiniCPM-V 4.6 就绪 (BF16 %s)", device)
        return {"model": model, "processor": proc, "kind": "transformers"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("[semantic] BF16 加载失败 (%s), 尝试 NF4", exc)
        return _load_nf4()


def _load_nf4():
    """备用: NF4 4bit 量化 (显存腾出后可 GPU 加速; CPU 上不如 BF16 快)."""
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoProcessor, BitsAndBytesConfig  # noqa: PLC0415
        from transformers.models.minicpmv4_6 import MiniCPMV4_6ForConditionalGeneration  # noqa: PLC0415

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        try:
            model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
                str(TF_DIR), quantization_config=quant,
                device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
        except Exception:  # noqa: BLE001
            model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
                str(TF_DIR), quantization_config=quant,
                device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True)
        proc = AutoProcessor.from_pretrained(str(TF_DIR), trust_remote_code=True)
        _loaded_at = time.time()
        logger.info("[semantic] MiniCPM-V 4.6 NF4 就绪 (设备=%s)", next(model.parameters()).device)
        return {"model": model, "processor": proc, "kind": "transformers"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("[semantic] NF4 加载失败: %s", exc)
        return None


def describe_image(image_b64: str, prompt: str = "用一两句话描述这张图片里发生了什么") -> str | None:
    """描述一张图 (base64 输入 → 中文描述)."""
    llm = _load()
    if llm is None:
        return None
    with _INFER_LOCK:
        try:
            if llm.get("kind") == "transformers":
                return _describe_transformers(llm, image_b64, prompt)
            # llama.cpp 多模态
            img_bytes = base64.b64decode(image_b64)
            out = llm.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=120,
                temperature=0.3,
            )
            text = (out.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return (text or "").strip()[:200] or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("[semantic] 推理失败: %s", exc)
            return None


def _describe_transformers(llm, image_b64: str, prompt: str) -> str | None:
    """transformers 推理 (MiniCPM: chat template 生成占位 → images+text 传入)."""
    from PIL import Image  # noqa: PLC0415
    import io  # noqa: PLC0415

    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    # 08-07 修复: MiniCPM-V 4.6 切片模式 (max_slice_nums=9) 对任意尺寸有 shape bug
    # (640x480 → 'shape [3,1082,1152] invalid for size 3741696')。
    # 预处理: 限制最大边 ≤ scale_resolution(448) → 单片模式, 避开切片路径;
    # 保持宽高比, 441/448 的 patch 对齐由处理器内部处理。
    max_side = 448
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)
    model, proc = llm["model"], llm["processor"]
    # 1. chat template 生成带 <|image_pad|> 的文本
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        add_generation_prompt=True, tokenize=False,
    )
    # 2. 处理器: images + text (模板含占位); inputs 搬到模型设备 (GPU 混合时)
    inputs = proc(text=text, images=[img], return_tensors="pt")
    try:
        dev = next(model.parameters()).device
        if str(dev) != "cpu":
            inputs = {k: v.to(dev) for k, v in inputs.items()}
    except Exception:  # noqa: BLE001
        pass
    out = model.generate(**inputs, max_new_tokens=40, temperature=0.3)  # 08-07: 80→40 提速 (~40%生成时间)
    reply = proc.decode(out[0], skip_special_tokens=True)
    # 净化: decode 含整个对话历史 (user+assistant), 只取 assistant 之后 + 去 think
    if "assistant" in reply:
        reply = reply.split("assistant", 1)[-1]
    reply = re.sub(r"<\|?(think|/think)\|?>\s*", "", reply).strip()
    if prompt in reply:
        reply = reply.split(prompt, 1)[-1]
    return (reply or "").strip()[:200] or None


def describe_frame(frame: np.ndarray, prompt: str = "") -> str | None:
    """从 numpy 帧 → base64 → 描述 (摄像头关键帧用)."""
    try:
        import cv2  # noqa: PLC0415

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode()
        return describe_image(b64, prompt or "用一两句话描述画面里的人在做什么、周围环境")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[semantic] 帧描述失败: %s", exc)
        return None
