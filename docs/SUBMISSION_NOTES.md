# Submission Notes

This artifact was prepared as a supplementary code package for the ACE-Compression paper.

## Included

- Benchmark evaluation code (launcher scripts, evaluation runners, custom task definitions)
- Compression / pruning scripts (SparseGPT one-shot pruning)
- vLLM model serving startup templates
- Configuration documentation (models, compression methods, benchmarks)
- Processed benchmark result files (canonical CSV outputs)
- Report aggregation utilities
- Result validation scripts
- Third-party asset and license documentation

## Excluded

- Model checkpoint weights and tokenizer files
- Original benchmark datasets (must be obtained from their sources)
- Unprocessed model output logs and raw evaluation outputs
- Local caches and temporary files
- PsychoBench questionnaires and source code
- Figure-generation scripts

## Root Sentinel

The file `configs/models.yaml` is used by packaged shell and Python entrypoints to find the artifact root without relying on the original machine path.

## Expected Model Layout

All packaged scripts assume checkpoints will be placed under `models/` inside this artifact tree. Because weights are excluded, users should either:

1. Place checkpoints under the `models/` tree, or
2. Create symlinks from the `models/` tree to actual checkpoint storage

The included `serving/*/start_vllm_*.sh` scripts are startup templates that reference these model paths.

## Package Scope

The artifact covers all 10 benchmarks and 6 model families evaluated in the paper, spanning four ability dimensions:

- Procedural reasoning: IFEval, GSM8K, HumanEval
- Knowledge grounding: TruthfulQA, C-Eval, RoleEval
- Social cognition: ToM-Bench, EQ-Bench
- Persona fidelity: CharacterEval, PsychoBench

The final paper model families: Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B, Llama3.1-8B, Llama3.1-70B, DeepSeek-V2-Lite-16B.
