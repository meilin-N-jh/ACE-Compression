#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
find_artifact_root() {
  local dir="$SCRIPT_DIR"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/configs/models.yaml" ]]; then
      printf "%s\n" "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(find_artifact_root || true)}"
if [[ -z "${ARTIFACT_ROOT}" ]]; then
  echo "Could not locate artifact root (configs/models.yaml)." >&2
  exit 1
fi

# =============================================================================
# 剪枝工具环境安装脚本
# 在 pruning 环境中安装 llmcompressor 和 SliceGPT
#
# 使用方法:
#     conda activate pruning
#     bash setup_compress_env.sh
# =============================================================================

set -e

echo "=============================================================================="
echo "安装剪枝工具依赖 - pruning 环境"
echo "=============================================================================="

# 1. 克隆 TransformerCompression (包含 SliceGPT)
echo ">>> 克隆 TransformerCompression 仓库..."
cd ${ARTIFACT_ROOT}/pruning_scripts
if [ ! -d "TransformerCompression" ]; then
    git clone https://github.com/microsoft/TransformerCompression.git
fi
cd TransformerCompression

# 2. 安装到 pruning 环境
echo ">>> 安装 TransformerCompression (包含 SliceGPT)..."
conda run -n pruning pip install -e .[experiment] --break-system-packages

# 3. 验证安装
echo ""
echo "=============================================================================="
echo "验证安装"
echo "=============================================================================="

conda run -n pruning python -c "import slicegpt; print('✓ SliceGPT installed successfully!')"
conda run -n pruning python -c "import llmcompressor; print(f'✓ llmcompressor {llmcompressor.__version__}')"

echo ""
echo "=============================================================================="
echo "✓ 剪枝工具依赖安装完成!"
echo "=============================================================================="
echo ""
echo "使用说明:"
echo "  1. 激活环境: conda activate pruning"
echo "  2. 2:4 稀疏剪枝: cd pruning_scripts/<model> && python run_2_4_sparse.py"
echo "  3. SliceGPT 剪枝: cd pruning_scripts/<model> && bash run_slicegpt.sh"
echo ""
