# 压缩模型评测调用运行指南

本文档说明在完成模型剪枝后，如何调用现有的 A 类和 B 类评测脚本。

---

## 一、剪枝模型目录结构

剪枝完成后，模型目录应如下：

```
models/
├── Qwen2.5-14B/
│   ├── Qwen2.5-14B-Instruct-FP16/     # 原 FP16
│   ├── Qwen2.5-14B-2to4-Sparse/       # 2:4 稀疏 (新增)
│   └── Qwen2.5-14B-SliceGPT-25/      # SliceGPT 25% (新增)
├── Qwen2.5-32B/
│   ├── Qwen2.5-32B-Instruct/
│   ├── Qwen2.5-32B-2to4-Sparse/      # 新增
│   └── Qwen2.5-32B-SliceGPT-25/      # 新增
├── DeepSeek-V2-Lite-Chat-16B/
│   ├── DeepSeek-V2-Lite-Chat/
│   ├── DeepSeek-V2-Lite-Chat-2to4-Sparse/      # 新增
│   └── DeepSeek-V2-Lite-Chat-SliceGPT-25/      # 新增
└── Llama3-70b/
    ├── Llama-3.1-70B-Instruct/
    ├── Llama-3.1-70B-Instruct-2to4-Sparse/      # 新增
    └── Llama-3.1-70B-Instruct-SliceGPT-25/      # 新增
```

---

## 二、端口分配表

| 模型 | 变体 | 端口 | GPU |
|------|------|------|-----|
| Qwen2.5-14B | FP16 | 8100 | 0 |
| Qwen2.5-14B | 2:4 Sparse | 8104 | 0 |
| Qwen2.5-14B | SliceGPT | 8105 | 0 |
| Qwen2.5-32B | FP16 | 8207 | 7 |
| Qwen2.5-32B | 2:4 Sparse | 8204 | 7 |
| Qwen2.5-32B | SliceGPT | 8205 | 7 |
| DeepSeek-V2-Lite | FP16 | 8300 | 6 |
| DeepSeek-V2-Lite | 2:4 Sparse | 8304 | 6 |
| DeepSeek-V2-Lite | SliceGPT | 8305 | 6 |
| Llama3-70b | FP16 | 8000 | 0,1 |
| Llama3-70b | 2:4 Sparse | 8004 | 0,1 |
| Llama3-70b | SliceGPT | 8005 | 0,1 |

---

## 三、评测流程

### 步骤 1: 启动 vLLM 服务

根据模型选择对应的启动脚本：

```bash
# 2:4 Sparse 模型
bash ${ARTIFACT_ROOT}/models/<model_family>/start_vllm_sparse.sh

# SliceGPT 模型
bash ${ARTIFACT_ROOT}/models/<model_family>/start_vllm_slicegpt.sh
```

验证服务启动：
```bash
curl http://127.0.0.1:<PORT>/v1/models
```

---

### 步骤 2: 运行 A 类评测 (lm-eval 直接运行)

A 类评测无需启动 vLLM 服务，直接使用 lm-eval 加载模型：

| Benchmark | 命令 |
|-----------|------|
| **C-eval** | `cd ${ARTIFACT_ROOT}/paper_benchmarks/05_C-eval/launchers/<model_folder> && bash run_fp16_vllm_ceval.sh` |
| **IFEval** | `cd ${ARTIFACT_ROOT}/paper_benchmarks/01_IFEval/launchers/<model_folder> && bash run_fp16_vllm_ifeval.sh` |
| **TruthfulQA** | `cd ${ARTIFACT_ROOT}/paper_benchmarks/04_TruthfulQA/launchers/<model_folder> && bash run_fp16_vllm_truthfulqa.sh` |
| **gsm8k** | `cd ${ARTIFACT_ROOT}/paper_benchmarks/02_gsm8k/launchers/<model_folder> && bash run_fp16_vllm_gsm8k.sh` |
| **human-eval** | `cd ${ARTIFACT_ROOT}/paper_benchmarks/03_human-eval/benchmark/<model_folder> && export HF_ALLOW_CODE_EVAL=1 && bash run_fp16_vllm_humaneval.sh` |

**注意**: 需要先为每个压缩模型创建对应的评测脚本文件夹，例如：
- `paper_benchmarks/05_C-eval/launchers/Qwen2.5-14B-2to4-Sparse/`
- `paper_benchmarks/05_C-eval/launchers/Qwen2.5-14B-SliceGPT-25/`

可以将现有的 `run_fp16_vllm_*.sh` 脚本复制并重命名，然后修改其中的模型路径和端口。

---

### 步骤 3: 运行 B 类评测 (需要 vLLM 服务)

B 类评测需要先启动 vLLM 服务（步骤 1），然后运行评测：

| Benchmark | 命令 |
|-----------|------|
| **role-eval** | `cd ${ARTIFACT_ROOT}/paper_benchmarks/06_roleeval/benchmark/<model_folder> && bash run.sh` |
| **CharacterEval** | `cd ${ARTIFACT_ROOT}/paper_benchmarks/09_CharacterEval/benchmark/<model_folder> && bash run_fp16.sh` |
| **ToM-Bench** | `cd ${ARTIFACT_ROOT}/paper_benchmarks/07_ToM-Bench/benchmark/<model_folder> && bash run_fp16.sh` |
| **PsychoBench** | `cd ${ARTIFACT_ROOT}/paper_benchmarks/10_PsychoBench/benchmark/<model_folder> && bash run_fp16.sh` |
| **eq-bench** | `cd ${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark/<model_folder> && bash run_fp16.sh` |

---

## 四、批量评测脚本模板

如果需要批量评测，可以参考以下模板：

### 4.1 A 类评测批量运行

```bash
#!/bin/bash
# 批量运行 A 类评测

MODEL_FAMILY=$1  # 如 Qwen2.5-14B
MODEL_VARIANT=$2  # 如 Qwen2.5-14B-2to4-Sparse

cd ${ARTIFACT_ROOT}/paper_benchmarks

# C-eval
cd C-eval/${MODEL_FAMILY}
cp run_fp16_vllm_ceval.sh run_${MODEL_VARIANT}_ceval.sh
sed -i "s|qwen2.5-14b-fp16|${MODEL_VARIANT}|g" run_${MODEL_VARIANT}_ceval.sh
LIMIT=10 bash run_${MODEL_VARIANT}_ceval.sh
```

---

## 五、结果汇总

评测完成后，在各 benchmark 文件夹下运行汇总脚本：

```bash
# C-eval
python ${ARTIFACT_ROOT}/scripts/sync_numeric_reports.py

# IFEval
python ${ARTIFACT_ROOT}/scripts/sync_numeric_reports.py

# gsm8k
python ${ARTIFACT_ROOT}/scripts/sync_gsm8k_reports.py
```

---

## 六、注意事项

1. **端口冲突**: 确保不同模型使用不同端口，参考上面的端口分配表
2. **GPU 显存**: 2:4 Sparse 和 SliceGPT 模型通常比 FP16 需要更少显存，可以适当调整 `--gpu-memory-utilization`
3. **Smoke Test**: 首次运行使用 `LIMIT=10` 进行验证
4. **离线模式**: 如需离线运行，添加：
   ```bash
   export TRANSFORMERS_OFFLINE=1
   export HF_DATASETS_OFFLINE=1
   ```
