# Assets

This table summarizes external models, datasets, benchmarks, and software referenced by the ACE-Compression supplementary package. The package does not redistribute model weights or original benchmark datasets. It includes ACE-Compression evaluation/compression code, helper scripts, configuration files, and processed evaluation results for research verification, and does not relicense any third-party assets.

Asset | Type | Source / URL | Version or commit | License / terms | How used | Included in supplement
--- | --- | --- | --- | --- | --- | ---
Qwen2.5 models | model | https://huggingface.co/Qwen | not logged | Apache-2.0 | evaluated as FP16 parent models and compressed variants where applicable | No model weights redistributed
Llama3.1 models | model | https://huggingface.co/meta-llama | not logged | Llama 3.1 Community License | evaluated as FP16 parent models and compressed variants where applicable | No model weights redistributed
DeepSeek-V2-Lite | model | https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite | not logged | DeepSeek License | evaluated as FP16 parent model and compressed variants where applicable | No model weights redistributed
vLLM | inference backend | https://github.com/vllm-project/vllm | not logged | Apache-2.0 | model serving / inference backend | No
SparseGPT | pruning method / software | https://github.com/IST-DASLab/sparsegpt | not logged | Apache-2.0 | SparseGPT-based pruning setup | No
bitsandbytes | quantization software | https://github.com/bitsandbytes-foundation/bitsandbytes | not logged | MIT | BnB 4-bit quantization | No
AWQ / llm-awq | quantization software | https://github.com/mit-han-lab/llm-awq | not logged | MIT | AWQ quantization | No
AutoGPTQ / GPTQ tooling | quantization software | https://github.com/AutoGPTQ/AutoGPTQ | not logged | MIT | GPTQ quantization where applicable | No
yahma/alpaca-cleaned | calibration dataset | https://huggingface.co/datasets/yahma/alpaca-cleaned | not logged | CC-BY-4.0 | pruning calibration samples | No original dataset redistributed; source link/configuration only.
IFEval | benchmark | https://huggingface.co/datasets/google/IFEval | not logged | Apache-2.0 | instruction-following evaluation | No original dataset redistributed; evaluation wrapper and processed results only
GSM8K / grade-school-math | benchmark | https://github.com/openai/grade-school-math | not logged | MIT | grade-school math reasoning evaluation | No original dataset redistributed; evaluation wrapper and processed results only
HumanEval | benchmark | https://github.com/openai/human-eval | not logged | MIT | code-generation evaluation | No original dataset redistributed; evaluation wrapper and processed results only
TruthfulQA | benchmark | https://github.com/sylinrl/TruthfulQA | not logged | Apache-2.0 | truthfulness / knowledge-grounding evaluation | No original dataset redistributed; evaluation wrapper and processed results only
C-Eval | benchmark | https://github.com/hkust-nlp/ceval | not logged | code MIT; dataset CC-BY-NC-SA-4.0 | Chinese exam-knowledge evaluation | No original dataset redistributed; evaluation wrapper and processed results only
RoleEval | benchmark dataset | https://www.scidb.cn/en/detail?dataSetId=c16b1553db0341d8ba0de71fe3e55a8c and https://arxiv.org/abs/2312.16132 | ScienceDB dataset page, published 2025-10-09 | dataset license CC BY-NC-SA 4.0 as shown on ScienceDB; arXiv paper license CC BY-SA 4.0 | bilingual role-grounded knowledge evaluation with RoleEval (en) and RoleEval (zh) slices | No original dataset redistributed; evaluation wrapper and processed results only
ToMBench | benchmark | https://github.com/zhchen18/ToMBench | not logged | MIT | social cognition / theory-of-mind evaluation | No original dataset redistributed; evaluation wrapper and processed results only
EQ-Bench | benchmark | https://github.com/EQ-bench/EQ-Bench | not logged | MIT | affective understanding / emotional intelligence evaluation | No original dataset redistributed; evaluation wrapper and processed results only
CharacterEval | benchmark | https://github.com/morecry/CharacterEval | not logged | MIT | role-play / persona evaluation | No original dataset redistributed; evaluation wrapper and processed results only
PsychoBench | benchmark | https://github.com/CUHK-ARISE/PsychoBench | not logged | GPL-3.0 | psychometric profile / persona evaluation | No original dataset redistributed; evaluation wrapper and processed results only
lm-evaluation-harness | evaluation framework | https://github.com/EleutherAI/lm-evaluation-harness
 | not logged | MIT | shared evaluation harness for supported benchmarks | third-party source code included where present under original license
