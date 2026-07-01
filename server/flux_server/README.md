# Flux 贴图生成服务

为 SAGE 场景生成提供地板/墙面贴图的 HTTP 服务。默认只占用**一块 GPU**，中央服务 **8090**，Worker **8091**（避免与 TRELLIS 8080/8081 冲突）。

## 1. 安装环境（一次性）

```bash
cd server/flux_server
chmod +x setup_flux_env.sh start_flux_server.sh
./setup_flux_env.sh
```

会创建名为 `flux` 的 Conda 环境（Python 3.10）并安装 PyTorch 2.4 + diffusers 等依赖。

## 2. Hugging Face

1. 在 https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev 同意模型协议  
2. 若尚未登录，执行一次 `huggingface-cli login`（已登录可跳过）

## 3. 启动服务

```bash
cd server/flux_server
./start_flux_server.sh
```

指定 GPU（默认 `0`）：

```bash
FLUX_GPU_ID=1 ./start_flux_server.sh
```

可选环境变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `CONDA_ENV_NAME` | `flux` | Conda 环境名 |
| `FLUX_GPU_ID` | `0` | 使用的物理 GPU 编号（仅此一块） |
| `FLUX_CENTRAL_PORT` | `8090` | 中央服务端口（SAGE 连接此端口） |
| `FLUX_WORKER_PORT` | `8091` | Worker 端口 |
| `STARTUP_TIMEOUT` | `900` | 等待模型加载超时（秒） |

## 4. 配置 SAGE Server

编辑 `server/key.json`：

```json
"FLUX_SERVER_URL": "http://localhost:8090"
```

材质后端默认已切换为 Flux（`server/constants.py` 中 `MATERIAL_BACKEND=flux`）。若需改回 MatFuse：

```bash
export MATERIAL_BACKEND=matfuse
```

## 5. 验证

```bash
curl http://localhost:8090/health
cd ../..
python server/floor_plan_materials/flux_generator.py
```

## 依赖版本

与 **PyTorch 2.4** 配套使用 `diffusers==0.32.2`（`diffusers>=0.33` 需更高版本 PyTorch，否则会 import 失败）。若环境曾 `pip install -U diffusers`，请重新执行 `./setup_flux_env.sh`。

## 显存

使用 `bfloat16` + `enable_model_cpu_offload()`，单卡约 **10–16 GB** 显存峰值；建议系统内存 ≥32 GB。
