# vision-plugin · 视觉插件

标准化视觉感知插件。**检测-语义两段式管线**：轻量行为检测（毫秒级）+ 重量语义分析 + 级联触发。

## 架构

```
A 层 realtime   MediaPipe 行为/人脸检测 (CPU 毫秒级, 不占显存)
                → 事件流: person_enter / motion_burst / motion_chaos / motion_settle
B 层 semantic   MiniCPM-V 语义分析 (3s 级, 全图描述 / 行为-对象元组)
Pipeline 级联   A 触发 B: 大幅动作 → 语义分析 → 记忆回调 → 视觉对象↔记忆检索关联
```

## 安装

```bash
pip install -e <本插件路径>\vision_plugin
pip install mediapipe numpy opencv-python transformers torch Pillow
```

> 模型不捆绑在插件内——用 `model_dir` 参数引用现成 MiniCPM-V 目录（或先跑无模型降级模式）。

## 跨环境使用（路径说明）

- 示例路径（`E:\abc` 等）是**本机演示**；插件用 `Path(__file__).resolve()` 推导自身位置，不依赖固定盘符
- **无模型时优雅降级**：MiniCPM 缺失 → 语义分析返回提示（不抛异常）；mediapipe 缺失 → 自动降级 cv2 背景差分（运动检测仍工作）；均无需改代码
- 推荐显式配置模型目录（避免每次探测）：
  - MCP 模式：设 `VISION_MODEL_DIR` / `VISION_MODEL_ROOT` 环境变量
  - 库模式：`VisionPipeline(model_dir=..., model_root=...)`
- 完整价值需配对记忆插件：`VisionPipeline(storage=memory_plugin_instance)` 才启用"视觉对象→记忆检索关联"；`storage=None` 时是纯检测工具（显式降级，不报错）

## MCP 接入（暴露为标准 MCP Server）

任意 MCP 客户端可通过 stdio 直接使用本插件的视觉理解能力（图片分析 / 画面快照 / 行为事件）：

```jsonc
// mcpServers 配置:
{ "command": "<你的python.exe绝对路径>",  # 例: E:/ai/venv/Scripts/python.exe
  "args": ["-m", "vision_plugin.mcp_server"] }
```

**暴露工具（3 个）**：`analyze_image`（本地图片路径 → MiniCPM-V 中文描述）/
`snapshot`（当前画面状态：人在场/运动强度）/ `get_events`（行为事件流：person_enter/motion_burst 等）

**环境变量**：`VISION_MODEL_DIR`（MiniCPM-V 模型目录）、`VISION_MODEL_ROOT`（mediapipe 模型根目录）；
**未配置时自动使用插件自带模型**；模型不可用时工具返回明确提示

依赖：`pip install fastmcp mcp`

## 快速开始

```python
import cv2
from vision_plugin import VisionPipeline

# 方式1: 不传模型参数 → 自动探测插件自带 models/ 目录 (开箱即用, 跨机器无需改路径)
pipe = VisionPipeline(storage=mem_plugin)

# 方式2: 自定义模型路径时显式指定 (例如放在其他目录)
# pipe = VisionPipeline(model_root="<你的模型根目录>",  # 含 models/mediapipe/
#                       model_dir="<你的minicpm目录>",  # 含 transformers/
#                       storage=mem_plugin)

# 摄像头循环: 逐帧喂入 → 检测 + 级联 (行为触发语义 → 自动记忆)
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    st = pipe.feed(frame)        # {"realtime", "semantic", "motion_level"}
    if st["semantic"]:
        print("本轮触发了语义分析")

# 主动分析一帧
desc = pipe.analyze(frame, "描述画面里的人在做什么")
```

## 事件模型（A 层 → 级联动作）

| 事件 | 触发条件 | 级联动作 |
|---|---|---|
| `person_enter` | 人脸从无到有 | B 全图场景描述（存记忆） |
| `motion_burst` | 运动从静到动（原始 motion>0.15） | B 行为-对象分析（"动作:xx 对象:xx"）+ 视觉对象↔记忆检索关联 |
| `motion_chaos` | 进入剧烈运动（motion>0.4，独立于人脸判断） | 降级记录"画面剧烈运动，无法分辨对象"，**不跑全图** |
| `motion_settle` | 剧烈运动结束 | 回溯补偿：重分析静止画面，绑定运动后的场景 |

设计要点：
- **行为-对象元组**：motion_burst 用专用 prompt 引导 VLM 输出 `动作:抓取 对象:苹果` 格式，存入记忆供"我刚才在干嘛"类问题召回
- **时间戳回溯绑定**：记忆存的是**事件触发时刻 T0**（非分析完成时刻），因果不错乱
- **运动幅度门控**：剧烈运动不硬分析（避免幻觉），事后补全——模拟人类"先感知动态，再聚焦物体"

## API 参考

### `VisionPipeline(model_root=None, model_dir=None, storage=None, sem_cooldown=10, chaos_cooldown=30, anchor_cooldown=1800)`

| 参数 | 默认 | 说明 |
|---|---|---|
| `model_root` | None（用 mediapipe 包内模型） | mediapipe .task 模型根目录（`{root}/models/mediapipe/`） |
| `model_dir` | `models/minicpm-v46` | MiniCPM-V 目录（含 `transformers/` 或 GGUF 文件） |
| `storage` | None | 记忆回调。**鸭子类型**：`remember(text, room, metadata, dedupe)` + 可选 `recall_with_scores`；None=只检测不记忆 |
| `sem_cooldown` | 10s | 语义分析冷却（B 分析 3s+，防事件风暴刷爆） |
| `chaos_cooldown` | 30s | 剧烈运动降级记录冷却 |
| `anchor_cooldown` | 1800s | 视觉对象↔记忆检索关联冷却（防打扰） |

### `feed(frame) -> dict`
- `frame`：BGR ndarray（cv2 读取的帧）
- 返回：`{"ok": True, "realtime": {faces, activity, motion_level, ...}, "semantic": bool, "motion_level": float}`
- `semantic=True` 表示本轮事件触发了 MiniCPM 分析并（可选）写记忆

### `analyze(frame, prompt="") -> str | None`
- 主动语义分析一帧；模型不可用时返回 None（首次打一次 warning）

### `snapshot() -> dict` / `pop_events(since_ts=0, limit=5) -> list[dict]`
- 实时状态快照 / 事件队列拉取

## 视觉对象↔记忆检索关联（视觉→记忆唤醒）

行为分析出对象 → 全库检索历史记忆 → **双通道命中**（对象字面共词 或 相似度 ≥0.60）→ 存 `source=anchor` 记忆：

```python
# 存储对象需提供 recall_with_scores(query, n_results, room)
# 命中示例: 画面出现"登山鞋" → 记忆里有"买了新登山鞋" → 关联命中 → 上层主动话:"你拿的登山鞋，是上次买的那个吗？"
```

> 已知限制：MiniLM 对中文短词区分度差（需要推理的关联如"药瓶↔胃痛"无法识别），采用"宁可漏报不误报"策略。换中文 embedding 后可解锁弱关联。

## 无模型降级模式

不配 `model_dir`/`model_root` 也能跑：
- MediaPipe 模型缺失 → 自动降级 cv2 背景差分（motion 检测仍工作，人脸检测不可用）
- MiniCPM 缺失 → 语义分析返回 None，检测/事件/降级记录链路照常

## 常见问题

**Q: MiniCPM 分析报 shape 错误？**
A: 图片尺寸需限制（MiniCPM-V 4.6 切片模式对超大图有 bug）。`describe_frame` 内部已做 448px 内预处理；若自传 base64 请走 `describe_image` 并控制分辨率。

**Q: 语义分析太频繁/太慢？**
A: 调大 `sem_cooldown`（默认 10s）；B 层推理受 CPU/GPU 限制，`max_new_tokens` 可自行调低（`semantic.py` 中）。

**Q: storage 需要什么接口？**
A: 只需 `remember(text, room="chat", metadata=None, dedupe=True) -> str`；要做视觉对象↔记忆检索关联再加 `recall_with_scores(query, n_results=5, room=None) -> list[dict]`。memory-plugin 完全满足。

