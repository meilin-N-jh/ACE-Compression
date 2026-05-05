# LLM 剪枝脚本

本目录包含用于对 FP16 全精度模型进行剪枝的脚本，支持两种剪枝方法：

1. **2:4 半结构化稀疏剪枝** - 使用 llmcompressor (SparseGPT)
2. **SliceGPT 结构化剪枝** - 使用 Microsoft TransformerCompression

---

## 支持的模型

| 模型 | FP16 源模型 | 2:4 Sparse 输出 | SliceGPT 输出 |
|------|-------------|-----------------|---------------|
| Qwen2.5-14B | `Qwen2.5-14B-Instruct-FP16` | `Qwen2.5-14B-2to4-Sparse` | `Qwen2.5-14B-SliceGPT-25` |
| Qwen2.5-32B | `Qwen2.5-32B-Instruct` | `Qwen2.5-32B-2to4-Sparse` | `Qwen2.5-32B-SliceGPT-25` |
| DeepSeek-V2-Lite | `DeepSeek-V2-Lite-Chat` | `DeepSeek-V2-Lite-Chat-2to4-Sparse` | `DeepSeek-V2-Lite-Chat-SliceGPT-25` |
| Llama3-70b | `Llama-3.1-70B-Instruct` | `Llama-3.1-70B-Instruct-2to4-Sparse` | `Llama-3.1-70B-Instruct-SliceGPT-25` |

---

## 环境准备

### 1. 创建 conda 环境

```bash
conda create -n pruning python=3.10
conda activate pruning
```

### 2. 安装依赖

```bash
# 方式一: 使用安装脚本
cd ${ARTIFACT_ROOT}/pruning_scripts
bash setup_compress_env.sh

# 方式二: 手动安装
conda activate pruning
pip install llmcompressor
cd ${ARTIFACT_ROOT}/pruning_scripts/TransformerCompression
pip install -e .[experiment]
```

**注意**: TransformerCompression 仓库已克隆到 `${ARTIFACT_ROOT}/pruning_scripts/TransformerCompression`

---

## 使用方法

### 2:4 半结构化稀疏剪枝

**原理**: 每 2 个参数中保留 1 个，零 50% 参数

```bash
# 激活环境
conda activate pruning

# 进入模型目录
cd ${ARTIFACT_ROOT}/pruning_scripts/Qwen2.5-14B

# 运行剪枝
python run_2_4_sparse.py
```

**配置说明** (可在脚本中修改):
- `NUM_SAMPLES`: 校准样本数 (默认 512)
- `SEQ_LEN`: 序列长度 (默认 2048)
- `SPARSITY`: 稀疏度 (默认 0.5 = 50%)

---

### SliceGPT 结构化剪枝

**原理**: 移除 25% 的神经元/隐藏维度

```bash
# 激活环境
conda activate pruning

# 进入模型目录
cd ${ARTIFACT_ROOT}/pruning_scripts/Qwen2.5-14B

# 运行剪枝
bash run_slicegpt.sh
```

**配置说明** (可在脚本中修改):
- `--sparsity 0.25`: 剪枝 25%
- `--cal-nsamples 128`: 校准样本数
- `--cal-batch-size 8`: 批大小

---

## 剪枝后流程

### 1. 启动 vLLM 服务

剪枝完成后，使用对应的 vLLM 启动脚本：

```bash
# 2:4 Sparse 模型
bash ${ARTIFACT_ROOT}/models/<model>/start_vllm_sparse.sh

# SliceGPT 模型
bash ${ARTIFACT_ROOT}/models/<model>/start_vllm_slicegpt.sh
```

### 2. 端口分配

| 模型 | 变体 | 端口 ||------|------|-----|
| Qwen2.5-14B GPU |
|------ | FP16 | 8100 | 0 |
| Qwen2.5-14B | 2:4 Sparse | 8104 | 0 |
| Qwen2.5-14B | SliceGPT | 8105 | 0 |
| Qwen2.5-32B | FP16 | 8200 | 7 |
| Qwen2.5-32B | 2:4 Sparse | 8204 | 7 |
| Qwen2.5-32B | SliceGPT | 8205 | 7 |
| DeepSeek-V2-Lite | FP16 | 8300 | 6 |
| DeepSeek-V2-Lite | 2:4 Sparse | 8304 | 6 |
| DeepSeek-V2-Lite | SliceGPT | 8305 | 6 |
| Llama3-70b | FP16 | 8000 | 0,1 |
| Llama3-70b | 2:4 Sparse | 8004 | 0,1 |
| Llama3-70b | SliceGPT | 8005 | 0,1 |

### 3. 运行评测

详细评测流程见 [run_compressed_evals_guide.md](./run_compressed_evals_guide.md)

---

## 目录结构

```
pruning_scripts/
├── README.md                          # 本文件
├── setup_compress_env.sh              # 环境安装脚本
├── run_compressed_evals_guide.md      # 评测指南
├── MODEL_PORT_PLAN.md                 # 端口规划
├── TransformerCompression/            # SliceGPT 仓库 (已安装)
├── Qwen2.5-14B/
│   ├── run_2_4_sparse.py             # 2:4 稀疏剪枝
│   └── run_slicegpt.sh               # SliceGPT 剪枝
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

## 常见问题

### Q1: 剪枝需要多少显存?

| 模型 | 2:4 Sparse | SliceGPT |
|------|------------|----------|
| 14B | ~16GB | ~20GB |
| 32B | ~40GB | ~48GB |
| 70b | ~80GB (2 GPU) | ~100GB (2 GPU) |

### Q2: 剪枝后模型精度下降明显吗?

- **2:4 Sparse**: 通常精度下降 1-3%
- **SliceGPT 25%**: 通常精度下降 2-5%

可通过微调恢复精度。

### Q3: 支持其他剪枝比例吗?

可以，修改脚本中的参数:
- 2:4 Sparse: 修改 `SPARSITY` 变量
- SliceGPT: 修改 `--sparsity` 参数

---

## 参考资料

- [llmcompressor 文档](https://llmcompressor.readthedocs.io/)
- [SliceGPT 论文](https://arxiv.org/abs/2401.06121)
- [TransformerCompression GitHub](https://github.com/microsoft/TransformerCompression)
