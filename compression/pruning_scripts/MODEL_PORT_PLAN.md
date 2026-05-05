# 模型端口分配与目录规划

本文档记录所有模型（含 FP16、量化、压缩变体）的端口分配和目录结构。

---

## 端口总览表

| 模型家族 | 变体 | 端口 | GPU | 模型目录 |
|----------|------|------|-----|----------|
| **Qwen2.5-14B** | | | | |
| | FP16 | 8100 | 0 | `Qwen2.5-14B-Instruct-FP16` |
| | BNB-4bit | 8101 | 1 | `Qwen2.5-14B-Instruct-BNB-4bit` |
| | AWG | 8102 | 2 | `Qwen2.5-14B-Instruct-AWQ` |
| | GPTQ-INT4 | 8103 | 3 | `Qwen2.5-14B-Instruct-GPTQ-Int4` |
| | **2:4 Sparse** | **8104** | 0 | `Qwen2.5-14B-2to4-Sparse` |
| | **SliceGPT-25%** | **8105** | 0 | `Qwen2.5-14B-SliceGPT-25` |
| **Qwen2.5-32B** | | | | |
| | FP16 | 8207 | 7 | `Qwen2.5-32B-Instruct` |
| | BNB-4bit | 8201 | - | `Qwen2.5-32B-Instruct-BNB-4bit` |
| | AWG | 8202 | - | `Qwen2.5-32B-Instruct-AWQ` |
| | GPTQ-INT4 | 8203 | - | `Qwen2.5-32B-Instruct-GPTQ-Int4` |
| | **2:4 Sparse** | **8204** | 7 | `Qwen2.5-32B-2to4-Sparse` |
| | **SliceGPT-25%** | **8205** | 7 | `Qwen2.5-32B-SliceGPT-25` |
| **DeepSeek-V2-Lite** | | | | |
| | FP16 | 8300 | 6 | `DeepSeek-V2-Lite-Chat` |
| | AWG | 8302 | - | `DeepSeek-V2-Lite-Chat-AWQ` |
| | GPTQ-INT4 | 8303 | - | `DeepSeek-V2-Lite-gptq-4bit` |
| | FP8 | - | - | `DeepSeek-V2-Lite-Chat-FP8` |
| | **2:4 Sparse** | **8304** | 6 | `DeepSeek-V2-Lite-Chat-2to4-Sparse` |
| | **SliceGPT-25%** | **8305** | 6 | `DeepSeek-V2-Lite-Chat-SliceGPT-25` |
| **Llama3-70b** | | | | |
| | FP16 | 8000 | 0,1 | `Llama-3.1-70B-Instruct` |
| | AWG-INT4 | - | - | `Llama-3.1-70B-Instruct-AWQ-INT4` |
| | GPTQ-INT4 | - | - | `Meta-Llama-3.1-70B-Instruct-GPTQ-INT4` |
| | GPTQ-INT8 | - | - | `Meta-Llama-3.1-70B-Instruct-GPTQ-INT8` |
| | LoRA | - | - | `Llama-3.1-70B-Instruct-LORA` |
| | **2:4 Sparse** | **8004** | 0,1 | `Llama-3.1-70B-Instruct-2to4-Sparse` |
| | **SliceGPT-25%** | **8005** | 0,1 | `Llama-3.1-70B-Instruct-SliceGPT-25` |

---

## 端口分配规则

- **十位**: 表示模型家族
  - 8x: Qwen/DeepSeek 系列
  - 7x: 预留
  - 6x: 预留
  - 0x: Llama

- **个位**: 表示变体类型
  - x000: FP16
  - x001: BNB-4bit
  - x002: AWG
  - x003: GPTQ-INT4
  - x004: **2:4 Sparse** (新增)
  - x005: **SliceGPT** (新增)

---

## 剪枝脚本位置

所有剪枝脚本存放在 `pruning_scripts/` 目录：

```
pruning_scripts/
├── Qwen2.5-14B/
│   ├── run_2_4_sparse.py      # 2:4 稀疏剪枝
│   └── run_slicegpt.sh        # SliceGPT 剪枝
├── Qwen2.5-32B/
│   ├── run_2_4_sparse.py
│   └── run_slicegpt.sh
├── DeepSeek-V2-Lite-Chat-16B/
│   ├── run_2_4_sparse.py
│   └── run_slicegpt.sh
└── Llama3-70b/
    ├── run_2_4_sparse.py
    └── run_slicegpt.sh
```

---

## vLLM 启动脚本位置

各模型目录下的 `start_vllm_*.sh` 脚本：

```
models/
├── Qwen2.5-14B/
│   ├── start_vllm_fp16.sh
│   ├── start_vllm_awq.sh
│   ├── start_vllm_gptq_int4.sh
│   ├── start_vllm_bnb_4bit.sh
│   ├── start_vllm_sparse.sh       # 2:4 Sparse
│   └── start_vllm_slicegpt.sh     # SliceGPT
├── ...
```

---

## 评测脚本位置

### A 类 (lm-eval)

```
paper_benchmarks/
├── 05_C-eval/launchers/<model_folder>/run_fp16_vllm_ceval.sh
├── 01_IFEval/launchers/<model_folder>/run_fp16_vllm_ifeval.sh
├── 04_TruthfulQA/launchers/<model_folder>/run_fp16_vllm_truthfulqa.sh
├── 02_gsm8k/launchers/<model_folder>/run_fp16_vllm_gsm8k.sh
└── 03_human-eval/benchmark/<model_folder>/run_fp16_vllm_humaneval.sh
```

### B 类 (vLLM API)

```
paper_benchmarks/
├── 06_roleeval/benchmark/<model_folder>/run.sh
├── 09_CharacterEval/benchmark/<model_folder>/run_fp16.sh
├── 07_ToM-Bench/benchmark/<model_folder>/run_fp16.sh
├── 10_PsychoBench/benchmark/<model_folder>/run_fp16.sh
└── 08_eq-bench/benchmark/<model_folder>/run_fp16.sh
```
