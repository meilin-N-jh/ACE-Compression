# EQ-Bench 评测完整指南

## 1. 项目概述

**EQ-Bench** 是用于评估大语言模型情商能力的中文基准测试框架。通过多单位选择题评估模型在情感管理、关系理解、社会认知等方面的表现。

### 核心特性

- **OpenAI Chat Completions API 集成**：支持 vLLM 兼容的 OpenAI API 服务
- **隔离式配置系统**：每个模型拥有独立的配置文件，互不干扰
- **多量化版本支持**：FP16、AWQ、GPTQ、BNB-4bit 等精度
- **直接结果输出**：结果自动保存到模型对应的 `results/` 文件夹
- **灵活的评测参数**：可配置迭代次数、问卷数量、输出格式等

---

## 2. 项目结构

```
paper_benchmarks/08_eq-bench/benchmark/
├── eq-bench.py                           # 主程序入口
├── config.cfg                            # 根配置（备用，不推荐使用）
├── EQBENCH_EVALUATION_GUIDE.md          # 本指南文件
├── lib/                                  # 核心库
│   ├── eq_bench_utils.py                # 工具函数
│   ├── run_bench.py                     # 评测引擎
│   ├── run_query.py                     # 查询执行（OpenAI API）
│   └── ...
├── instruction-templates/                # 提示模板
│   ├── OpenAIChat.yaml                  # OpenAI 格式模板（推荐）
│   ├── Qwen-ChatML.yaml                 # Qwen 内部格式
│   └── ...（其他模板）
│
├── data/                                 # 评测数据集
│   └── ...
│
└── <model_folder>/                      # 各模型的评测目录
    ├── config/                          # 隔离的配置文件
    │   ├── config_fp16.cfg              # FP16 精度配置
    │   ├── config_awq.cfg               # AWQ 量化配置
    │   ├── config_bnb_4bit.cfg          # BNB 4bit 配置
    │   ├── config_gptq_int4.cfg         # GPTQ INT4 配置
    │   └── ...
    │
    ├── run_fp16.sh                      # FP16 评测脚本
    ├── run_awq.sh                       # AWQ 评测脚本
    ├── run_gptq_int4.sh                 # GPTQ 评测脚本
    ├── run_bnb_4bit.sh                  # BNB 4bit 评测脚本
    │
    ├── results/                         # 结果输出目录
    │   ├── raw_results_*.json           # 原始回答数据
    │   └── benchmark_results_*.csv      # 评测结果汇总
    │
    ├── logs/                            # 运行日志
    │   └── eqbench_*.log
    │
    └── eqbench_results_summary.csv      # 结果摘要
```

### 当前支持的模型

| 模型名称 | 模型文件夹 | 默认端口 | 支持配置 |
|---------|-----------|--------|--------|
| Qwen2.5-14B | qwen2.5-14b | 8100 | FP16, AWQ, GPTQ, BNB-4bit, 剪枝版本 |
| Qwen2.5-32B | qwen2.5-32b | 8200 | FP16, AWQ, GPTQ, BNB-4bit, 剪枝版本 |
| Qwen2.5-7B | qwen2.5-7b | 8300 | FP16, AWQ, GPTQ, BNB-4bit |
| Qwen1-7B | qwen1 | 8400 | FP16, AWQ, GPTQ |
| DeepSeek-V2-Lite-16B | deepseek-v2-lite-16b | 8500 | FP16, AWQ, GPTQ, BNB-4bit |
| Llama3.1-8B | llama3.1-8b | 8600 | FP16, AWQ, GPTQ, BNB-4bit |
| Llama3.1-70B | llama70B | 8700 | FP16, AWQ, GPTQ, BNB-4bit |

---

## 3. 快速开始

### 3.1 前置准备

#### 环境激活
```bash
conda activate benchmark
```

#### 配置 CUDA 可见设备（可选）
# 启动 vLLM OpenAI API 服务
bash ${ARTIFACT_ROOT}/models/DeepSeek-V2-Lite-Chat-16B/start_vllm_fp16.sh &
export CUDA_VISIBLE_DEVICES=0  # 只使用GPU 0
```

### 3.2 基本流程

#### 第一步：启动 vLLM 服务

以 **DeepSeek-V2-Lite-16B FP16** 为例：

```bash
# 启动 vLLM OpenAI API 服务
bash ${ARTIFACT_ROOT}/models/DeepSeek-V2-Lite-Chat-16B/start_vllm_fp16.sh
```

**重要**：
- 该脚本会在后台启动 vLLM 服务
- 默认监听地址：`http://127.0.0.1:8500/v1/`
- 首次启动会进行模型加载，可能需要 5-10 分钟
- 启动成功标志：可以访问 `http://127.0.0.1:8500/v1/models` 并获得模型列表

#### 验证服务（可选）
```bash
# 在另一个终端检查服务
curl -s http://127.0.0.1:8500/v1/models | jq .

# 应该能看到类似的输出：
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "deepseek-v2-lite-fp16",
cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/deepseek-v2-lite-16b
#       "owned_by": "vllm",
#       ...
#     }
#   ]
# }
```

#### 第二步：运行评测

```bash
# 进入模型评测目录
cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/deepseek-v2-lite-16b

# 运行 FP16 评测
bash run_fp16.sh
```

**脚本会自动**：
1. 检查 vLLM 服务是否已启动
2. 加载配置文件：`config/config_fp16.cfg`
3. 执行评测程序
4. 将结果保存到 `results/` 文件夹
5. 生成日志到 `logs/` 文件夹

#### 第三步：查看结果

评测完成后，结果文件会保存在模型的 `results/` 目录：

```bash
# 进入结果目录
cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/deepseek-v2-lite-16b/results/

# 查看原始回答（JSON 格式）
cat raw_results_*.json | jq .

# 查看最终评测结果（CSV 格式）
cat benchmark_results_*.csv
```

---

## 4. 详细配置说明

### 4.1 配置文件结构

每个模型的配置文件位于 `<model_folder>/config/config_<variant>.cfg`

**示例：deepseek-v2-lite-16b/config/config_fp16.cfg**

```ini
[OpenAI]
# vLLM OpenAI 兼容服务的 URL（必须）
openai_compatible_url = http://127.0.0.1:8500/v1/
# API 密钥（对本地 vLLM 可设为 EMPTY）
api_key = EMPTY

[Huggingface]
access_token =
cache_dir =

[Results upload]
google_spreadsheet_url =

[Creative Writing Benchmark]
judge_model_api = local
judge_model = ${ARTIFACT_ROOT}/models/Llama3-70b/Llama-3.1-70B-Instruct
judge_model_api_key =

[Options]
trust_remote_code = true

[Oobabooga config]
ooba_launch_script = oobabooga/server.py
ooba_params_global = --trust-remote-code --nowebui --model-dir ${ARTIFACT_ROOT}/models/Llama3-70b --attn-implementation eager
automatically_launch_ooba = false
ooba_request_timeout = 600

[Benchmarks to run]
# 格式：benchmark_id, template_name, model_id, quantization, load_in_bits, iteration, inference_engine, api_key, base_url
# 
# 关键字段说明：
# - benchmark_id: 唯一标识（自定义命名）
# - template_name: 提示词模板（推荐 OpenAIChat）
# - model_id: 模型标识（与 vLLM 注册的模型 ID 相同）
# - quantization: 量化方法（none/awq/gptq/...)
# - load_in_bits: 加载精度（8/4/...）
# - iteration: 评测迭代次数（推荐 10）
# - inference_engine: 推理引擎（openai/local/）
# - api_key: API 密钥（openai 引擎使用）
# - base_url: API 基础 URL（openai 引擎使用）
#
deepseek_v2_lite_fp16, OpenAIChat, deepseek-v2-lite-fp16, , none, 10, openai, , 
```

### 4.2 关键配置参数

#### [Benchmarks to run] 部分

| 参数 | 说明 | 示例 | 备注 |
|-----|------|------|------|
| benchmark_id | 评测唯一标识 | `deepseek_v2_lite_fp16` | 自定义，用于输出文件命名 |
| template_name | 提示词模板 | `OpenAIChat` | **必须为 OpenAIChat**（与engine=openai搭配） |
| model_id | 模型ID | `deepseek-v2-lite-fp16` | 需与 vLLM 注册的模型 ID 相同 |
| quantization | 量化方法 | `none` / `awq` / `gptq` / `bitsandbytes` | 仅作标记，实际量化由 vLLM 处理 |
| load_in_bits | 加载精度 | `none` / `8` / `4` / ... | 仅作标记 |
| iteration | 迭代次数 | `10` | 评测会运行该次数，取平均分 |
| inference_engine | 推理引擎 | `openai` | **必须为 openai**（使用 Chat Completions API） |
| api_key | API 密钥 | 留空 | 本地 vLLM 留空 |
| base_url | 基础 URL | 留空 | 使用 [OpenAI] 部分的 openai_compatible_url |

### 4.3 修改配置的常见场景

#### 场景 1：修改评测迭代次数

```ini
# 原配置（10次）
deepseek_v2_lite_fp16, OpenAIChat, deepseek-v2-lite-fp16, , none, 10, openai, , 

# 改为 5 次（快速测试）
deepseek_v2_lite_fp16, OpenAIChat, deepseek-v2-lite-fp16, , none, 5, openai, , 

# 改为 20 次（更精确的评分）
deepseek_v2_lite_fp16, OpenAIChat, deepseek-v2-lite-fp16, , none, 20, openai, , 
```

#### 场景 2：修改 API 服务地址

```ini
# 本地服务（localhost）
openai_compatible_url = http://127.0.0.1:8500/v1/

# 远程服务（需要修改为实际地址）
openai_compatible_url = http://192.168.1.100:8500/v1/

# 阿里云服务
openai_compatible_url = https://api.aliyun.com/v1/
api_key = your-api-key-here
```

#### 场景 3：添加多个模型并行评测

```ini
[Benchmarks to run]
# 模型 1：DeepSeek-V2-Lite-16B FP16
deepseek_v2_lite_fp16, OpenAIChat, deepseek-v2-lite-fp16, , none, 10, openai, , 

# 模型 2：DeepSeek-V2-Lite-16B AWQ
deepseek_v2_lite_awq, OpenAIChat, deepseek-v2-lite-awq, awq, 4, 10, openai, , 

# 模型 3：DeepSeek-V2-Lite-16B GPTQ（同一 vLLM 服务）
# 注意：量化版本需要在同一端口运行不同的模型配置
deepseek_v2_lite_gptq, OpenAIChat, deepseek-v2-lite-gptq, gptq, 4, 10, openai, , 
```

---

## 5. 运行指南

### 5.1 单个模型评测

#### 步骤 1：启动对应的 vLLM 服务

```bash
# 以 Qwen2.5-14B FP16 为例
bash ${ARTIFACT_ROOT}/models/Qwen2.5-14B/start_vllm_fp16.sh &

# 等待服务启动（约 5-10 分钟）
sleep 30
curl -s http://127.0.0.1:8100/v1/models
```

#### 步骤 2：运行对应的评测脚本

```bash
cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/qwen2.5-14b
bash run_fp16.sh
```

### 5.2 多个模型评测（顺序）

#### 方法 1：手动运行（推荐）

适合需要观察中间过程或 GPU 显存有限的情况。

```bash
# 模型 1：Qwen2.5-14B FP16
bash ${ARTIFACT_ROOT}/models/Qwen2.5-14B/start_vllm_fp16.sh &
sleep 30
cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/qwen2.5-14b
bash run_fp16.sh
# 评测完成后，停止 vLLM 服务
pkill -f "vllm serve"
sleep 10

# 模型 2：Qwen2.5-14B AWQ
bash ${ARTIFACT_ROOT}/models/Qwen2.5-14B/start_vllm_awq.sh &
sleep 30
bash run_awq.sh
pkill -f "vllm serve"
sleep 10

# 模型 3：Qwen2.5-32B FP16
bash ${ARTIFACT_ROOT}/models/Qwen2.5-32B/start_vllm_fp16.sh &
sleep 30
cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/qwen2.5-32b
bash run_fp16.sh
pkill -f "vllm serve"
```

#### 方法 2：批量脚本（仅当 GPU 足够时）

```bash
#!/bin/bash
# batch_eqbench.sh - 批量评测脚本

MODELS=(
    "qwen2.5-14b:fp16:8100"
    "qwen2.5-14b:awq:8100"
    "qwen2.5-32b:fp16:8200"
    "deepseek-v2-lite-16b:fp16:8300"
)

for model_config in "${MODELS[@]}"; do
    IFS=':' read -r model variant port <<< "$model_config"
    
    echo "[$(date)] Starting $model $variant on port $port..."
    bash ${ARTIFACT_ROOT}/models/$model/start_vllm_${variant}.sh &
    sleep 30
    
    cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/$model
    bash run_${variant}.sh
    
    echo "[$(date)] Finished $model $variant. Stopping vLLM..."
    pkill -f "vllm serve"
    sleep 10
done

echo "[$(date)] All evaluations completed!"
```

运行批量脚本：
```bash
bash batch_eqbench.sh
```

### 5.3 快速测试（Smoke Test）

评测完整轮次可能需要 1-2 小时。如果要快速验证配置是否正确，可以修改配置的 `iteration` 参数为 1：

```ini
[Benchmarks to run]
# 快速测试：只运行 1 次迭代
qwen25_14b_fp16, OpenAIChat, qwen2.5-14b-fp16, , none, 1, openai, , 
```

然后运行脚本：
```bash
bash run_fp16.sh
```

这样会在 5-10 分钟内完成评测，用于检验环境配置。

---

## 6. 结果说明

### 6.1 输出文件

评测完成后，结果会保存到 `<model_folder>/results/` 目录：

```
results/
├── raw_results_<timestamp>.json      # 原始回答数据
├── benchmark_results_<timestamp>.csv # 最终评测结果
└── ...（多次运行会产生多个文件）
```

### 6.2 结果文件格式

#### raw_results_*.json（原始数据）

```json
{
  "qwen25_14b_fp16_1": {
    "question_id": "Q1",
    "question": "小李和小王是同事。小李最近升职了，但小王没有...",
    "options": {
      "A": "祝贺小李升职",
      "B": "抱怨不公平",
      "C": "什么都不说",
      "D": "换工作"
    },
    "correct_answer": "A",
    "model_answer": "A",
    "is_correct": true,
    "model_response": "A",
    "timestamp": "2026-03-11T10:24:30"
  },
  ...
}
```

#### benchmark_results_*.csv（评测汇总）

```csv
Benchmark,Correct Answers,Total Questions,Accuracy (%)
qwen25_14b_fp16,1,1,100.00
```

### 6.3 查看结果

```bash
# 进入结果目录
cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/qwen2.5-14b/results/

# 查看最新的 CSV 结果
tail -1 benchmark_results_*.csv

# 使用 Python 查看详细的 JSON 结果
python3 -c "
import json
with open('raw_results_*.json') as f:
    data = json.load(f)
    for key, val in data.items():
        print(f'{key}: {val[\"is_correct\"]}')"

# 或使用 jq 工具
cat raw_results_*.json | jq '.[] | {question, model_answer, correct_answer, is_correct}'
```

---

## 7. 常见使用场景

### 场景 A：评估单个模型的多个量化版本

**目标**：对比 DeepSeek-V2-Lite-16B 的 FP16、AWQ、GPTQ、BNB-4bit 版本性能

```bash
#!/bin/bash
cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/deepseek-v2-lite-16b

for variant in fp16 awq gptq_int4 bnb_4bit; do
    echo \"===== Testing $variant =====\"
    
    # 启动服务
    bash ${ARTIFACT_ROOT}/models/DeepSeek-V2-Lite-Chat-16B/start_vllm_${variant}.sh &
    sleep 30
    
    # 运行评测
    bash run_${variant}.sh
    
    # 停止服务
    pkill -f "vllm serve"
    sleep 10
done

echo "Comparison complete!"
```

### 场景 B：评估多个不同的模型

**目标**：对比 Qwen2.5、Llama3.1、DeepSeek 三个模型系列

```bash
MODELS=(
    "qwen2.5-14b"
    "qwen2.5-32b"
    "llama3.1-8b"
    "llama70B"
    "deepseek-v2-lite-16b"
)

for model in "${MODELS[@]}"; do
    echo "===== Evaluating $model ====="
    
    # 启动对应的 vLLM 服务
    bash ${ARTIFACT_ROOT}/models/$model/start_vllm_fp16.sh &
    sleep 30
    
    # 运行评测
    cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/$model
    bash run_fp16.sh
    
    # 停止服务
    pkill -f "vllm serve"
    sleep 10
done
```

### 场景 C：自定义评测参数

**目标**：使用自定义参数运行特定评测

```bash
cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/deepseek-v2-lite-16b

# 修改配置（例如增加迭代次数）
sed -i 's/, 10, openai,/, 20, openai,/' config/config_fp16.cfg

# 启动服务和运行评测
bash ${ARTIFACT_ROOT}/models/DeepSeek-V2-Lite-Chat-16B/start_vllm_fp16.sh &
sleep 30
bash run_fp16.sh

# 恢复原配置
sed -i 's/, 20, openai,/, 10, openai,/' config/config_fp16.cfg
pkill -f \"vllm serve\"
```

---

## 8. 故障排查

### 问题 1：vLLM 服务连接失败

**错误信息**：
```
[ERROR] vLLM API server not found on port 8500
```

**解决方案**：

1. 检查服务是否启动
```bash
ps aux | grep vllm
curl -v http://127.0.0.1:8500/v1/models
```

2. 手动启动服务（带输出）
```bash
cd ${ARTIFACT_ROOT}/models/DeepSeek-V2-Lite-Chat-16B
bash start_vllm_fp16.sh  # 不加 & 让日志可见
```

3. 检查 GPU 状态
```bash
nvidia-smi
# 确保有足够的显存（Qwen2.5-14B FP16 需要约 28GB）
```

4. 检查端口占用
```bash
lsof -i :8100
# 如果被占用，先 kill 旧进程
kill -9 <PID>
```

### 问题 2：配置文件找不到或格式错误

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'config/config_fp16.cfg'
```

**解决方案**：

1. 确保在正确的目录运行（应在 `<model_folder>/` 下）
```bash
pwd  # 应输出 .../eq-bench/qwen2.5-14b
```

2. 检查配置文件是否存在
```bash
ls -la config/
```

3. 直接指定完整路径运行
```bash
cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark
EQBENCH_OUTPUT_DIR="./qwen2.5-14b/results" \
python -u eq-bench.py --config "qwen2.5-14b/config/config_fp16.cfg"
```

### 问题 3：API 调用返回 400 或 401 错误

**错误信息**：
```
openai.error.APIError: HTTP status 400 or 401
```

**解决方案**：

1. 检查 API URL 是否正确
```bash
# 在配置文件中验证
grep "openai_compatible_url" config/config_fp16.cfg
```

2. 手动测试 API
```bash
curl -X POST http://127.0.0.1:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-14b-fp16",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

3. 检查 vLLM 服务日志
```bash
# 查看 vLLM 启动日志（在 vLLM 启动的终端）
# 或查看是否有日志文件
tail -100 /tmp/vllm_*.log 2>/dev/null || echo "No log found"
```

### 问题 4：模型回答格式异常，无法解析答案

**错误信息**：
```
ValueError: Cannot parse model answer
```

**解决方案**：

1. 检查配置是否使用 OpenAIChat 模板
```bash
grep "template_name" config/config_fp16.cfg
# 应输出：OpenAIChat
```

2. 验证模型是否能正确返回 A/B/C/D 答案
```bash
# 手动测试
curl -X POST http://127.0.0.1:8500/v1/chat/completions \
    -H \"Content-Type: application/json\" \
    -d '{
        \"model\": \"deepseek-v2-lite-fp16\",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "请只回答 A、B、C 或 D。\n问题：这是一个测试。\nA. 选项A\nB. 选项B\nC. 选项C\nD. 选项D"}
    ],
    "max_tokens": 5
  }'
```

3. 检查是否需要调整 prompt 的指令
```bash
# 可能需要修改 data/eqbench.yaml 中的 prompt 格式
grep -A5 "prompt:" paper_benchmarks/08_eq-bench/benchmark/data/*.yaml
```

### 问题 5：结果文件为空或未生成

**现象**：运行完成但没有看到结果文件

**解决方案**：

1. 检查 RESULTS_DIR 是否正确设置
```bash
# 在脚本中添加调试输出
echo "Results directory: $EQBENCH_OUTPUT_DIR"
ls -la deepseek-v2-lite-16b/results/
```

2. 查看程序日志
```bash
cat deepseek-v2-lite-16b/logs/eqbench_fp16_*.log | tail -100
```

3. 手动运行并查看输出
```bash
cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark
EQBENCH_OUTPUT_DIR="./test_output" python -u eq-bench.py --config "deepseek-v2-lite-16b/config/config_fp16.cfg"
ls -la test_output/
```

---

## 9. 性能优化建议

### 9.1 加速评测

1. **减少迭代次数**（用于快速测试）
```ini
iteration = 5  # 从 10 改为 5
```

2. **并行运行多个 GPU**（如果有多张 GPU）
```bash
# 使用不同的 GPU 和端口
CUDA_VISIBLE_DEVICES=0 bash models/DeepSeek-V2-Lite-Chat-16B/start_vllm_fp16.sh &
CUDA_VISIBLE_DEVICES=1 bash models/Qwen2.5-14B/start_vllm_fp16.sh &

# 在不同终端运行对应评测
cd paper_benchmarks/08_eq-bench/benchmark/deepseek-v2-lite-16b && bash run_fp16.sh &
cd paper_benchmarks/08_eq-bench/benchmark/qwen2.5-14b && bash run_fp16.sh &
```

3. **调整 vLLM 显存利用率**
```bash
# 在 start_vllm_*.sh 中修改
# 默认：--gpu-memory-utilization 0.9
# 改为：--gpu-memory-utilization 0.95  # 更积极的显存使用
```

### 9.2 降低显存占用

1. **使用量化版本**
```bash
# FP16: 28GB (Qwen2.5-14B)
# AWQ/GPTQ: 8GB (Qwen2.5-14B)
# BNB-4bit: 12GB (Qwen2.5-14B)

bash models/Qwen2.5-14B/start_vllm_awq.sh    # 推荐显存有限时使用
```

3. **使用量化版本**
```bash
# DeepSeek-V2-Lite-16B FP16: 32GB
# DeepSeek-V2-Lite-16B AWQ: 8GB
# DeepSeek-V2-Lite-16B GPTQ: 8GB

cd paper_benchmarks/08_eq-bench/benchmark/deepseek-v2-lite-16b
bash run_awq.sh
```

---

## 10. 结果汇总与对比

### 10.1 生成结果摘要

评测完成后，可以生成模型结果摘要：

```bash
# 方法 1：查看最新结果
cat ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/qwen2.5-14b/results/benchmark_results_*.csv | tail -1

# 方法 2：汇总所有模型结果
cat > summarize_eqbench.py << 'EOF'
import os
import csv
from pathlib import Path

results = {}
base_dir = "${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark"

for model_dir in sorted(os.listdir(base_dir)):
    model_path = os.path.join(base_dir, model_dir)
    if not os.path.isdir(model_path) or model_dir in ['data', 'lib', 'reports', 'tools', 'instruction-templates']:
        continue
    
    results_path = os.path.join(model_path, 'results')
    if not os.path.exists(results_path):
        continue
    
    # 找最新的 benchmark_results 文件
    csv_files = sorted(Path(results_path).glob('benchmark_results_*.csv'))
    if csv_files:
        latest_csv = csv_files[-1]
        with open(latest_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[f"{model_dir}_{row['Benchmark']}"] = row['Accuracy (%)']

# 输出结果
print("Model Evaluation Results")
print("=" * 50)
for model, accuracy in sorted(results.items()):
    print(f"{model:40s}: {accuracy}")
EOF

python3 summarize_eqbench.py
```

### 10.2 对比多个模型

```bash
# 生成对比表格
python3 << 'EOF'
import os
import csv
from pathlib import Path
from collections import defaultdict

# 统计各模型精度
results = defaultdict(list)
base_dir = "${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark"

for model_dir in sorted(os.listdir(base_dir)):
    model_path = os.path.join(base_dir, model_dir)
    results_path = os.path.join(model_path, 'results')
    
    if not os.path.exists(results_path):
        continue
    
    csv_files = sorted(Path(results_path).glob('benchmark_results_*.csv'))
    if csv_files:
        latest_csv = csv_files[-1]
        with open(latest_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                acc = float(row['Accuracy (%)'].strip('%'))
                results[model_dir].append((row['Benchmark'], acc))

# 输出表格
print("\nEQ-Bench Evaluation Results")
print("=" * 80)
print(f"{'Model':<25} {'Benchmark':<30} {'Accuracy':<15}")
print("-" * 80)

for model in sorted(results.keys()):
    for bench, acc in results[model]:
        print(f"{model:<25} {bench:<30} {acc:>6.2f}%")

print("-" * 80)
EOF
```

---

## 11. 注意事项

### 安全与最佳实践

1. **不要修改根目录的 config.cfg**
   - 每个模型都有自己的 config 文件
   - 直接使用 `--config` 参数指定模型配置

2. **定期检查 GPU 显存**
   ```bash
   nvidia-smi -l 1  # 每秒查看一次
   ```

3. **vLLM 服务后台运行**
   - 建议在后台运行（加 `&`）
   - 但首次部署时建议前台运行（观察启动过程）

4. **结果备份**
   - 评测完成后，备份 `results/` 文件夹
   ```bash
    cp -r deepseek-v2-lite-16b/results deepseek-v2-lite-16b/results_backup_$(date +%Y%m%d)
   ```

5. **长时间运行注意事项**
   - 评测可能需要 1-2 小时
   - 建议使用 `screen` 或 `tmux` 在远程环境运行
   ```bash
   screen -S eqbench
   # 然后运行评测脚本
   bash run_fp16.sh
   # Ctrl+A, D 退出 screen（保持运行）
   # screen -r eqbench 重新连接
   ```

### 常见错误避免

| 错误 | 原因 | 解决 |
|-----|------|------|
| 找不到 config 文件 | 运行路径错误 | 确保在 `<model_folder>/` 下运行 |
| vLLM 连接失败 | 服务未启动 | 提前启动 vLLM 服务 |
| 模型文件不存在 | 模型未下载 | 检查 `${ARTIFACT_ROOT}/models/` |
| 显存不足 | GPU 内存不够 | 使用量化版本或较小模型 |
| 结果文件为空 | 程序异常中止 | 查看日志文件排查错误 |

---

## 12. 完整示例：从零开始评测

这是一个完整的、从零开始的评测流程示例。

### 步骤 1：准备环境

```bash
# 激活环境
conda activate benchmark

# 进入 EQ-Bench 目录
cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark
```

### 步骤 2：启动 vLLM 服务

```bash
# 在后台启动（推荐）
bash ${ARTIFACT_ROOT}/models/DeepSeek-V2-Lite-Chat-16B/start_vllm_fp16.sh &

# 或前台启动（第一次调试时推荐）
# bash ${ARTIFACT_ROOT}/models/DeepSeek-V2-Lite-Chat-16B/start_vllm_fp16.sh

# 等待服务启动（约 5-10 分钟）
echo "Waiting for vLLM to start..."
sleep 30

# 验证服务
curl -s http://127.0.0.1:8500/v1/models | jq .
```

### 步骤 3：运行评测

```bash
# 进入模型目录
cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/deepseek-v2-lite-16b

# 运行评测（约 30 分钟 - 2 小时，取决于迭代次数）
bash run_fp16.sh
```

### 步骤 4：查看结果

```bash
# 查看原始数据
ls -lh results/
cat results/raw_results_*.json | jq . | head -50

# 查看最终结果
cat results/benchmark_results_*.csv
```

### 步骤 5：清理资源

```bash
# 停止 vLLM 服务
pkill -f "vllm serve"

# 或更温柔地停止
kill %1  # 如果在当前 shell 后台运行
```

### 完整脚本（一键运行）

将以下内容保存为 `quick_eval.sh`：

```bash
#!/bin/bash
set -euo pipefail

# 配置
MODEL_DIR="deepseek-v2-lite-16b"
VARIANT="fp16"
VLLM_PORT="8500"
VLLM_BIN="${ARTIFACT_ROOT}/models/DeepSeek-V2-Lite-Chat-16B/start_vllm_${VARIANT}.sh"
EQBENCH_DIR="${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark"

# 激活环境
conda activate benchmark

echo "===== Step 1: Starting vLLM Service ====="
bash "$VLLM_BIN" &
VLLM_PID=$!
sleep 30

# 检查服务
if ! curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM service failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

echo "✓ vLLM service is ready"

echo ""
echo "===== Step 2: Running EQ-Bench ====="
cd "$EQBENCH_DIR/$MODEL_DIR"
bash run_${VARIANT}.sh

# 获取结果
echo ""
echo "===== Step 3: Results ====="
LATEST_CSV=$(ls -t results/benchmark_results_*.csv | head -1)
if [ -f "$LATEST_CSV" ]; then
    echo "Latest results:"
    cat "$LATEST_CSV"
else
    echo "WARNING: No results found"
fi

echo ""
echo "===== Step 4: Cleanup ====="
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm serve" || true

echo "✓ Evaluation completed!"
```

运行：
```bash
chmod +x quick_eval.sh
./quick_eval.sh
```

---

## 附录 A：端口与模型映射

| 模型 | 标准端口 | FP16 | AWQ | GPTQ | BNB-4bit | 说明 |
|-----|--------|------|-----|------|----------|-----|
| Qwen2.5-14B | 8100 | 8100 | 8100 | 8100 | 8100 | 需按顺序启动不同精度版本 |
| Qwen2.5-32B | 8200 | 8200 | 8200 | 8200 | 8200 | - |
| Qwen2.5-7B | 8300 | 8300 | 8300 | 8300 | 8300 | - |
| Qwen1-7B | 8400 | 8400 | 8400 | 8400 | - | 仅支持 3 种精度 |
| DeepSeek-V2-Lite-16B | 8500 | 8500 | 8500 | 8500 | 8500 | - |
| Llama3.1-8B | 8600 | 8600 | 8600 | 8600 | 8600 | - |
| Llama3.1-70B | 8700 | 8700 | 8700 | 8700 | 8700 | - |

---

## 附录 B：配置文件完整模板

```ini
[OpenAI]
openai_compatible_url = http://127.0.0.1:8100/v1/
api_key = EMPTY

[Huggingface]
access_token =
cache_dir =

[Results upload]
google_spreadsheet_url =

[Creative Writing Benchmark]
judge_model_api = local
judge_model = ${ARTIFACT_ROOT}/models/Llama3-70b/Llama-3.1-70B-Instruct
judge_model_api_key =

[Options]
trust_remote_code = true

[Oobabooga config]
ooba_launch_script = oobabooga/server.py
ooba_params_global = --trust-remote-code --nowebui --model-dir ${ARTIFACT_ROOT}/models/Llama3-70b --attn-implementation eager
automatically_launch_ooba = false
ooba_request_timeout = 600

[Benchmarks to run]
# <benchmark_id>, <template_name>, <model_id>, <quantization>, <load_in_bits>, <iteration>, <inference_engine>, <api_key>, <base_url>
benchmark_name, OpenAIChat, model-id, none, none, 10, openai, , 
```

---

## 附录 C：参考链接

- **vLLM 官方文档**：https://docs.vllm.ai/
- **OpenAI API 文档**：https://platform.openai.com/docs/api-reference/chat/create
- **模型下载**：${ARTIFACT_ROOT}/models/
- **评测数据**：${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/data/

---

**最后更新**：2026-03-11  
**作者**：LLM Ability Test Team  
**版本**：1.0
