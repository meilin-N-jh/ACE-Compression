# EQ-Bench for Qwen2.5-14B

## 📋 脚本列表

- `run_fp16.sh` - FP16模型评测
- `run_bnb_4bit.sh` - BNB 4bit模型评测
- `run_awq.sh` - AWQ模型评测
- `run_gptq_int4.sh` - GPTQ INT4模型评测
- `run_trim.sh` - Trim模型评测

## 🚀 使用方法

### 1. 激活环境

```bash
conda activate benchmark
```

### 2. 运行评测（选择一个）

```bash
cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/qwen2.5-14b

# FP16
bash run_fp16.sh

# BNB 4bit
bash run_bnb_4bit.sh

# AWQ
bash run_awq.sh

# GPTQ INT4
bash run_gptq_int4.sh

# Trim
bash run_trim.sh
```

### 脚本功能

每个脚本会自动：
1. ✅ 启动vLLM服务器（后台运行）
2. ✅ 等待服务器就绪
3. ✅ 运行EQ-Bench评测
4. ✅ 保存结果到 `results/` 目录
5. ✅ 自动停止vLLM服务器

## 📁 结果文件

- **原始结果**: `results/raw_results_<model>.json`
- **评测结果**: `results/benchmark_results_<model>.csv`
- **日志**: `logs/eqbench_<model>_<timestamp>.log`

## ⚙️ 配置文件

- `config/config_fp16.cfg` - FP16配置
- `config/config_bnb_4bit.cfg` - BNB 4bit配置
- `config/config_awq.cfg` - AWQ配置
- `config/config_gptq_int4.cfg` - GPTQ INT4配置
- `config/config_trim.cfg` - Trim配置

每个配置文件指定对应的vLLM服务器端口（8100-8104）。
