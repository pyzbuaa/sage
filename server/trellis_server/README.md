# TRELLIS 3D 生成服务

本项目将 Microsoft TRELLIS 封装为 HTTP 服务，支持文本生成 GLB 模型。启动脚本在 `TRELLIS_GPU_IDS` 指定的每张 GPU 上启动一个 Worker（默认 GPU 4–7），并在 `8080` 端口启动统一的轮询分发服务。

环境安装与服务启动已经分离：

- `setup_trellis_env.sh`：一次性下载 TRELLIS 并向已有 Conda 环境安装依赖。
- `start_trellis_server.sh`：只校验环境、加载模型和启动服务，不安装 Conda、不创建环境、不重复安装依赖。
- `requirements-trellis.txt`：Python 依赖列表。

## 1. 前期准备

### 1.1 硬件与驱动

需要使用配备 NVIDIA GPU 的 Linux 服务器。当前依赖使用 PyTorch 2.4.0 CUDA 12.4 构建版本，执行以下命令确认驱动和 GPU 可见：

```bash
nvidia-smi
```

部分依赖需要编译 CUDA 扩展，因此系统还需要可用的 CUDA Toolkit（包括 `nvcc`）以及 C/C++ 编译工具。

### 1.2 系统依赖

Ubuntu/Debian 可执行：

```bash
sudo apt-get update
sudo apt-get install -y git build-essential ninja-build libx11-6 libgl1 libglib2.0-0
```

这些系统包不会由项目脚本自动安装。

### 1.3 Conda 环境

服务器需要预先安装 Conda。本项目不会安装 Miniconda，也不会执行 `conda init` 或修改 shell 配置。

默认复用名为 `trellis` 的 Conda 环境，并要求 Python 3.10。先检查环境是否存在：

```bash
conda env list
conda run -n trellis python --version
```

如果还没有该环境，只需在首次部署时手动创建一次：

```bash
conda create -n trellis python=3.10 -y
```

也可以复用其他环境，后续命令统一设置环境名：

```bash
export CONDA_ENV_NAME=my_trellis
```

脚本通过 `conda run` 定位环境，不要求提前执行 `conda activate`。

### 1.4 Hugging Face 访问令牌

服务首次启动时会下载 `microsoft/TRELLIS-text-xlarge` 模型。准备一个具备模型读取权限的 Hugging Face Token，并在启动服务的终端中设置：

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

请勿把 Token 写入脚本或提交到代码仓库。首次加载需要访问 Hugging Face，并占用相应的模型缓存磁盘空间；后续启动会复用缓存。

## 2. 安装

进入本项目目录后运行一次：

```bash
chmod +x setup_trellis_env.sh start_trellis_server.sh
CONDA_ENV_NAME=trellis ./setup_trellis_env.sh
```

安装脚本会：

1. 检查 Conda 和指定环境是否存在，并确认其使用 Python 3.10。
2. 将 TRELLIS 克隆到当前项目下的 `TRELLIS/`；源码已存在时只初始化或补齐子模块。
3. 将 PyTorch、TRELLIS 运行依赖和 HTTP 服务依赖安装到指定的已有 Conda 环境。

默认目录可以通过环境变量覆盖：

```bash
TRELLIS_DIR=/data/models/TRELLIS \
CONDA_ENV_NAME=trellis \
./setup_trellis_env.sh
```

安装需要访问 GitHub、PyPI、PyTorch 和 NVIDIA Kaolin 软件源。如果安装 CUDA 扩展失败，优先检查 `nvcc --version`、编译器版本和 CUDA 环境变量。

## 3. 启动服务

前台启动：

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
CONDA_ENV_NAME=trellis ./start_trellis_server.sh
```

如果 TRELLIS 位于自定义目录，安装和启动必须使用相同的路径：

```bash
TRELLIS_DIR=/data/models/TRELLIS \
CONDA_ENV_NAME=trellis \
./start_trellis_server.sh
```

启动过程中会为 `TRELLIS_GPU_IDS` 中的每张 GPU 各加载一份模型。默认使用物理 GPU **4、5、6、7**（4 个 Worker，端口 `8081`–`8084`），中央服务端口 **8080**：

```bash
export TRELLIS_GPU_IDS=4,5,6,7
CONDA_ENV_NAME=trellis ./start_trellis_server.sh
```

按需覆盖卡号或端口：

```bash
TRELLIS_GPU_IDS=0,1,2,3 \
TRELLIS_CENTRAL_PORT=8080 \
TRELLIS_WORKER_BASE_PORT=8081 \
CONDA_ENV_NAME=trellis ./start_trellis_server.sh
```

若只使用单卡，显式指定一张即可：

```bash
TRELLIS_GPU_IDS=0 CONDA_ENV_NAME=trellis ./start_trellis_server.sh
```

**与 SAGE Server 并发对齐**：SAGE 侧 `place_objects_in_room` 默认 `TRELLIS_GENERATION_MAX_WORKERS=4`（`server/constants.py`）。该值应 **≤ Worker 数量**；超过 Worker 数会在同卡上叠加推理导致 OOM。

```bash
export TRELLIS_GENERATION_MAX_WORKERS=4   # 在运行 layout Server / Client 前设置
```

脚本会等待所有 Worker 健康检查通过，默认最多等待 600 秒，可按需调整：

```bash
STARTUP_TIMEOUT=1200 ./start_trellis_server.sh
```

按 `Ctrl+C` 停止服务时，脚本会清理它启动的 Worker 进程。

## 4. 访问服务

服务监听所有网络接口的 `8080` 端口。本机地址：

```text
http://127.0.0.1:8080
```

局域网客户端应将下文的 `SERVER_IP` 替换为服务器地址，可使用以下命令查看：

```bash
hostname -I
```

常用接口如下：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/` | 服务信息与接口导航 |
| `GET` | `/health` | 服务及 Worker 健康状态 |
| `GET` | `/metrics` | 系统、GPU 和任务指标 |
| `GET` | `/docs` | 浏览器接口文档 |
| `GET` | `/openapi.json` | OpenAPI 描述文件 |
| `GET` | `/api/v1/models` | 可用模型列表 |
| `POST` | `/generate` | 提交文本生成任务 |
| `GET` | `/job/<job_id>` | 查询任务或下载 GLB |

浏览器文档地址：

```text
http://SERVER_IP:8080/docs
```

健康检查示例：

```bash
curl http://SERVER_IP:8080/health
```

## 5. 生成并下载 GLB

### 5.1 提交任务

```bash
curl -X POST http://SERVER_IP:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"input_text":"a small red sports car"}'
```

服务立即返回 HTTP `202`，例如：

```json
{
  "status": "accepted",
  "job_id": "central_1710000000000_1234",
  "message": "Request received and processing started"
}
```

### 5.2 查询任务

将返回的任务编号替换到 URL 中：

```bash
curl -i http://SERVER_IP:8080/job/central_1710000000000_1234
```

- HTTP `202`：仍在生成，稍后继续查询。
- HTTP `200`：响应内容就是 GLB 文件。
- HTTP `404`：任务不存在，常见于任务编号错误或服务已经重启。
- HTTP `500`：生成失败，响应 JSON 中包含错误信息。

### 5.3 Python 完整示例

下面的客户端会提交任务、轮询状态并把结果保存为 `model.glb`：

```python
import time
import requests

base_url = "http://SERVER_IP:8080"

response = requests.post(
    f"{base_url}/generate",
    json={"input_text": "a small red sports car"},
    timeout=30,
)
response.raise_for_status()
job_id = response.json()["job_id"]
print("job_id:", job_id)

while True:
    response = requests.get(f"{base_url}/job/{job_id}", timeout=30)

    if response.status_code == 202:
        print("generating...")
        time.sleep(2)
        continue

    response.raise_for_status()
    with open("model.glb", "wb") as file:
        file.write(response.content)
    print("saved: model.glb")
    break
```

## 6. 运维与安全说明

- 任务状态和生成结果保存在进程内存中，服务重启后旧的 `job_id` 会失效。
- 每张 GPU 会加载一个模型实例（约 21GB 显存），请确保显存和主机内存充足；**每张卡同时只应处理 1 个生成任务**。
- SAGE Client 并发请求数（`TRELLIS_GENERATION_MAX_WORKERS`）应与 Worker 数量一致，避免多任务压到同一张卡。
- 首次启动通常比后续启动慢，因为需要下载并加载模型。
- 当前服务启用了 CORS，但没有身份认证、限流和 TLS。不要直接暴露到公网；生产环境应在前面增加 Nginx 等反向代理，并配置 HTTPS、认证、请求大小限制和访问控制。
- 若需要远程访问，请只开放中央服务的 `8080` 端口。Worker 端口建议通过防火墙限制为本机访问。
