# SAGE 场景生成 — 中文部署指南

Agent 驱动的室内场景生成：Client（Qwen3-VL Agent）通过 MCP 调用 Server（布局 / 材质 / 物体放置），3D 物体由 **TRELLIS** 生成，地板/墙贴图默认走 **Flux**。

## 架构一览

```
Client (client/)          Server (server/layout_wo_robot.py)     外部服务
  Qwen3-VL Agent    ←MCP→   布局 / 材质 / 放物体逻辑      →  LLM/VLM API (server/key.json)
                                                          →  TRELLIS (3D 物体)
                                                          →  Flux (地板/墙贴图)
                                                          →  Isaac Sim (可选，物理 Critic)
```

**最小可跑组合**：`simgen` 环境 + Server LLM API + TRELLIS + Flux + Client Qwen API。**不需要 MatFuse**（`MATERIAL_BACKEND=flux` 时）。

---

## 0. 前置条件

- Linux + NVIDIA GPU（建议 ≥24GB 显存；Flux 单独占一块卡）
- Conda（Miniforge / Mamba 均可）
- 能访问 Hugging Face（或提前下载模型到 `~/.cache/huggingface`）
- 本机或内网可连：**TRELLIS HTTP 服务**、**Flux HTTP 服务**、**LLM/VLM API**

---

## 1. 主环境 `simgen`（Client + Server 共用）

```bash
cd /path/to/sage
conda env create -f environment.yaml    # 环境名 simgen
conda activate simgen
```

**额外编译**（物体网格处理需要，见 `environment.yaml` 注释）：

- `pytorch3d`（匹配 torch 2.5.1+cu124）
- `nvdiffrast`

**预下载 CLIP / SBERT**（避免 Server 每次启动联网失败）：

```bash
export HTTPS_PROXY=http://127.0.0.1:7890   # 如需代理
unset HF_ENDPOINT                          # 不要用 hf-mirror 下 laion CLIP
# 下载完成后：
export HF_HUB_OFFLINE=1
```

权重缓存到 `~/.cache/huggingface/hub/` 后，Server 启动会快很多（仍须在 CPU 上加载约 2GB，约 3–5 分钟/次冷启动）。

---

## 2. Flux 贴图服务（默认材质后端）

独立 Conda 环境 `flux`，与 `simgen` 分离：

```bash
cd server/flux_server
chmod +x setup_flux_env.sh start_flux_server.sh
./setup_flux_env.sh          # 一次性
huggingface-cli login        # 同意 FLUX.1-Krea-dev 协议后登录
./start_flux_server.sh       # 中央 8090，Worker 8091，默认 GPU 0
```

验证：`curl http://localhost:8090/health`

在 `server/key.json` 中配置：

```json
"FLUX_SERVER_URL": "http://localhost:8090"
```

---

## 3. TRELLIS 3D 物体服务

独立部署（可与本机不同机器）。详见 `server/trellis_server/README.md`。

```bash
cd server/trellis_server          # 请用此目录下的脚本，不要用已废弃的 server/start_trellis_server.sh（已转发到同一路径）
./setup_trellis_env.sh       # 一次性
./start_trellis_server.sh    # 默认 8080 中央分发 + GPU 4/5/6/7 共 4 Worker
```

启动日志应出现：`TRELLIS_GPU_IDS=4,5,6,7 -> 4 worker(s)`。若看到 8 个 Worker，说明仍在跑旧进程或误用了会遍历全部 GPU 的旧脚本；先 `pkill -f 'server.py --port'` 再重启。

**推荐：4 卡部署（GPU 4/5/6/7）**

TRELLIS-text-xlarge 单卡约占 **21GB** 显存，每张卡只能同时跑 **1** 个生成任务。默认启动脚本会在物理 GPU **4、5、6、7** 上各起一个 Worker（端口 `8081`–`8084`），中央服务监听 **8080** 并轮询分发：

```bash
# 默认即 4,5,6,7；可按机器改卡号
export TRELLIS_GPU_IDS=4,5,6,7
CONDA_ENV_NAME=trellis ./start_trellis_server.sh
```

**Client 并发须与 Worker 数量一致（默认 4）**

`place_objects_in_room` 会并发向 TRELLIS 提交多个物体的 3D 生成请求。Server 侧默认 `TRELLIS_GENERATION_MAX_WORKERS=4`（见 `server/constants.py`）。若 Worker 只有 1 张卡却开 8 路并发，会在同一张 GPU 上叠加多个 `pipeline.run()`，极易 **CUDA OOM** 或 200 秒超时后重试雪崩。

```bash
# 与 TRELLIS Worker 数量保持一致（4 卡 → 4）
export TRELLIS_GENERATION_MAX_WORKERS=4
```

在 `server/key.json` 中配置：

```json
"TRELLIS_SERVER_URL": "http://<TRELLIS主机IP>:8080"
```

每个放置的物体都会请求 TRELLIS 生成 GLB，是全流程最耗时的环节之一。

---

## 4. Server 配置 `server/key.json`

```json
{
  "ANTHROPIC_API_KEY": "sk-...",
  "API_TOKEN": "你的API密钥",
  "API_URL_QWEN": "https://api.example.com/v1",
  "API_URL_OPENAI": "https://api.example.com/v1",
  "MODEL_DICT": {
    "claude": "claude-sonnet-4-20250514",
    "openai": "gpt-oss-120b",
    "qwen": "qwen3-vl-30b-a3b-thinking"
  },
  "TRELLIS_SERVER_URL": "http://172.23.x.x:8080",
  "FLUX_SERVER_URL": "http://localhost:8090"
}
```

- **Server 侧 LLM/VLM**：布局推理、`generate_scene_requirements`、门窗描述等走 `API_URL_OPENAI` / `ANTHROPIC_API_KEY`。
- `API_TOKEN` 应为 **密钥字符串**，不是 URL。

可选环境变量（启动 Server 前 export，或在 Client 启动子进程时传入）：

```bash
export MATERIAL_BACKEND=flux          # 默认 flux；matfuse 需另装整套 MatFuse
export PHYSICS_CRITIC_ENABLED=false   # 无 Isaac Sim 时建议关闭
export SEMANTIC_CRITIC_ENABLED=false
export HF_HUB_OFFLINE=1
export TRELLIS_GENERATION_MAX_WORKERS=4  # 与 TRELLIS Worker GPU 数一致，避免 OOM
export VLN_MODE=true                  # 默认 true：面向 VLN 的精简场景（详见 §11）
```

> **面向 VLN 的精简生成（默认开启）**：`VLN_MODE=true` 时只生成大件、可指代的地标家具 + 少量显著物体，不再"填满每个货架/桌面"，可显著减少 TRELLIS 生成物体数、大幅加速，并保持地面通行空间。详见 §11。

---

## 5. Client 配置 `client/key.json`

```json
{
  "API_TOKEN": "sk-...",
  "API_URL_QWEN": "https://api.example.com/v1",
  "MODEL_NAME": "qwen3-vl-32b-thinking"
}
```

- Client Agent 使用 **OpenAI 兼容接口** 调用 Qwen3-VL。
- `max_tokens` 须 ≤32768（网关限制）。
- 使用 thinking 模型时，Client 已配置 `tool_choice=auto`，并解析 `reasoning` 字段中的 tool call。

---

## 6. Isaac Sim 安装与启动

生成场景本身**不强制**需要 Isaac；但以下用途依赖 Isaac：

- 物理 / 语义 Critic（生成过程中自动摆物体、查碰撞）
- **导出 USD**（在 Isaac Sim 中打开、仿真、渲染）
- 机器人任务数据（Isaac Lab 管线）

### 6.1 安装

1. 安装 [Isaac Sim 4.2](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html)，解压到本机路径（如 `/data/users/pyz/isaacsim-4.2`）
2. 软链 SAGE 自带 MCP 扩展：

```bash
ln -s $(realpath server/isaacsim/isaac.sim.mcp_extension) \
      /path/to/isaacsim-4.2/exts/isaac.sim.mcp_extension
```

3. 编辑 `client/isaac_sim_conda.sh` 中的 `ISAACSIM_PATH`、`CONDA_ENV_NAME`（默认 `simgen`）

### 6.2 启动（导出 / Critic 前必须运行）

```bash
conda activate simgen
cd client
./isaac_sim_conda.sh --no-window --enable isaac.sim.mcp_extension
```

日志出现 `Isaac Sim MCP server started on localhost:8766` 表示就绪。无显示器服务器可加 `--no-window`。

后台常驻（与 `generate_from_room_desc.sh` 相同思路）：

```bash
while true; do
  ./client/isaac_sim_conda.sh --no-window --enable isaac.sim.mcp_extension
  sleep 5
done &
```

使用 Critic 时生成命令改为 `export PHYSICS_CRITIC_ENABLED=true`（语义 Critic 同理）。

---

## 7. 启动与生成

### 7.1 确认服务就绪

| 服务 | 检查命令 |
|------|----------|
| Flux | `curl http://localhost:8090/health` |
| TRELLIS | `curl http://<host>:8080/health` |
| Server 单独测 | 见下方 |

```bash
conda activate simgen
cd server
export HF_HUB_OFFLINE=1 PHYSICS_CRITIC_ENABLED=false SEMANTIC_CRITIC_ENABLED=false
python -c "import layout_wo_robot; print('Server import OK')"
```

### 7.2 运行场景生成

```bash
conda activate simgen
cd client

export HF_HUB_OFFLINE=1
export PHYSICS_CRITIC_ENABLED=false
export SEMANTIC_CRITIC_ENABLED=false

python client_generation_room_desc.py \
  --room_desc "a bedroom" \
  --server_paths ../server/layout_wo_robot.py
```

流程概要：

1. Client 连 MCP Server（冷启动会加载 CLIP/SBERT）
2. Qwen Agent 调用 `generate_room_layout`（房间结构 + Flux 材质 + 门窗）
3. 多轮 `place_objects_in_room`（TRELLIS 逐物体生成 3D 并摆放）

输出目录：`server/results/<layout_id>/`；任务记录：`client/room_descs/`。

### 7.3 脚本入口

`client/scripts/generate_from_room_desc.sh` 含 Isaac 后台循环，路径/环境名需按本机修改；**推荐直接用上面的 `python client_generation_room_desc.py` 命令**。

---

## 8. 导出到 Isaac Sim（USD / GLB）

完成场景生成后，在 `server/results/<layout_id>/` 下会有布局 JSON、材质贴图、物体网格等。导出前请确认 **Isaac Sim 已启动**（见 §6.2）。

### 8.1 获取 layout_id

- 终端日志：`🔍 Layout ID: layout_xxxx`
- 或查看 `server/results/` 下最新目录名
- 布局文件：`server/results/<layout_id>/<layout_id>.json`

### 8.2 方式 A：直接导出 USD（推荐）

通过 MCP 扩展，将布局转为**分物体 USD** 集合：

```bash
conda activate simgen
cd server/isaaclab
```

编辑 `save_layouts_to_usd.py` 末尾的 `layout_id = "..."`，然后：

```bash
python save_layouts_to_usd.py
```

输出：

| 路径 | 内容 |
|------|------|
| `server/results/<layout_id>/<layout_id>_usd_collection/` | 各墙/地/门窗/物体 USD |
| `.../layout_info.json` | 布局元数据 |
| `server/results/<layout_id>/<layout_id>.glb` | 整场景 GLB（便于预览） |

在 Isaac Sim 中：**File → Open** 打开 `_usd_collection` 下任意 USD，或整目录作为场景引用。

底层 API：`isaacsim.isaac_mcp.server.get_room_layout_scene_usd_separate_from_layout(layout_json, usd_dir)`。

### 8.3 方式 B：打包发布包（SAGE-10k 工具链）

用于批量导出、渲染或与 [SAGE-10k kits](https://huggingface.co/datasets/nvidia/SAGE-10k/tree/main/kits) 配合：

```bash
conda activate simgen
cd server
python pack_scene_to_zip.py --layout_id <layout_id> --upload_name <layout_id>
```

生成 `server/results/<layout_id>/<layout_id>.zip`（含布局 JSON、物体网格、材质、预览图等）。

解压后，用 HuggingFace 上的 **kits** 脚本进一步导出 GLB / USD / 渲染图（见 `client/README.md` §2.3）。

### 8.4 导出常见问题

| 现象 | 处理 |
|------|------|
| 连接 Isaac 失败 | 确认 `isaac_sim_conda.sh` 已运行，端口 **8766** 可达 |
| 扩展未找到 | 检查 `isaac.sim.mcp_extension` 软链 |
| USD 目录为空 | 确认 `<layout_id>.json` 存在且含 `objects` |
| 无显示器环境 | 启动加 `--no-window` |

---

## 9. 常见问题

| 现象 | 处理 |
|------|------|
| CLIP 下载 `Network unreachable` | 设代理 + `unset HF_ENDPOINT`；或预下载后 `HF_HUB_OFFLINE=1` |
| MatFuse / Pylette 报错 | Flux 模式不应再触发；确认 `MATERIAL_BACKEND=flux` |
| `max_tokens` 400 错误 | Client 中 `max_tokens` ≤ 32768 |
| `No response from Qwen3-VL`、0 tool call | 检查 API 模型名、`tool_choice`；thinking 模型需较久等待 |
| 每步都很慢 / 物体太多 | 每个物体都要 TRELLIS 生成（1–5 分钟）。默认已开启 `VLN_MODE=true` 精简物体数（§11）；仍慢可调小 `VLN_MAX_OBJECTS` |
| TRELLIS `CUDA out of memory` | 单卡只能 1 路生成；用 `TRELLIS_GPU_IDS=4,5,6,7` 起 4 Worker，并设 `TRELLIS_GENERATION_MAX_WORKERS=4` |
| TRELLIS `Timeout ... 200 seconds` | 队列过长或单卡过载；减并发或加 GPU Worker；单物体生成常需 1–5 分钟 |
| Server 每次启动 5 分钟 | CLIP/SBERT 冷加载；可保持 Server 进程常驻或后续改懒加载 |

---

## 10. 目录说明

| 路径 | 作用 |
|------|------|
| `client/` | Qwen Agent、生成脚本、`isaac_sim_conda.sh` |
| `server/` | MCP Server、布局/材质/物体逻辑 |
| `server/isaacsim/` | Isaac Sim MCP 扩展与 Python 桥接 |
| `server/isaaclab/` | USD/GLB 导出脚本（`save_layouts_to_usd.py`） |
| `server/flux_server/` | Flux 贴图 HTTP 服务 |
| `server/trellis_server/` | TRELLIS 3D HTTP 服务 |
| `matfuse-sd/` | 可选材质后端（`MATERIAL_BACKEND=matfuse`） |
| `environment.yaml` | `simgen` Conda 环境定义 |

更细的子模块说明见 `client/README.md`、`server/README.md`、`server/isaacsim/README.md`、`server/flux_server/README.md`。

---

## 11. 面向 VLN 的精简场景生成（VLN_MODE）

**动机**：全流程耗时 ≈ 物体数量 × 单物体 TRELLIS 生成时间（1–5 分钟/物体）。默认的居家真实感 prompt 会要求"每个货架 >5 件、每个桌面 ≥2 件"，导致小物件爆炸、生成极慢。面向 **VLN（视觉语言导航）** 的场景其实只需要**可通行空间 + 可用语言指代的大件地标家具**，不需要堆满小物。

**开关**（`server/constants.py` 与 `client/client_generation_room_desc.py` 共用，默认开启）：

```bash
export VLN_MODE=true               # 开启精简模式（默认）；=false 回退到原"填满真实感"行为
export VLN_MAX_OBJECTS=12          # 单房间物体提案总数上限
export VLN_MAX_ONTOP_PER_SURFACE=1 # 每个家具表面最多摆几个显著物体（0 = 表面不放小物）
```

启动生成时会由 Client 自动把这几个变量转发给 Server 子进程，无需重复 export。

**开启后的行为变化**：

| 环节 | 默认 (`VLN_MODE=true`) | 关闭 (`VLN_MODE=false`) |
|------|------------------------|--------------------------|
| 场景需求提案数 | 约 8–`VLN_MAX_OBJECTS` 个地标物 | ≥20 个，含大量小物 |
| 货架/桌面 | 保持整洁，每表面 ≤`VLN_MAX_ONTOP_PER_SURFACE` | 货架 >5 件、每表面 ≥2 件 |
| 代码兜底裁剪 | 超限提案自动裁掉 | 不裁剪 |
| Client 放置轮数 `max_tool_calls` | 8 | 15 |
| 完成判据 | 地标家具齐全 + 地面可通行即停 | 填满所有表面才停 |

**想更快**：把 `VLN_MAX_OBJECTS` 调小（如 6–8），并设 `VLN_MAX_ONTOP_PER_SURFACE=0`。
**想更丰富**：调大 `VLN_MAX_OBJECTS`，或直接 `VLN_MODE=false` 回退原行为。
