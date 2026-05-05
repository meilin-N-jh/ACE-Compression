#!/usr/bin/env python3
"""Generate isolated EQ-Bench run folders for vLLM + OpenAI Chat Completions workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OPENAI_CHAT_TEMPLATE = """template_type: openai_chat_messages
system_message: ""
user_template: "<|user-message|>"
"""

CONFIG_TEMPLATE = """[OpenAI]
openai_compatible_url = http://127.0.0.1:{port}/v1
api_key = EMPTY

[Huggingface]
access_token =
cache_dir =

[Results upload]
google_spreadsheet_url =

[Creative Writing Benchmark]
judge_model_api = local
judge_model =
judge_model_api_key =

[Options]
trust_remote_code = true

[Oobabooga config]
ooba_launch_script =
ooba_params_global =
automatically_launch_ooba = false
ooba_request_timeout = 600

[Benchmarks to run]
{run_id}, OpenAIChat, {model_id}, , none, {iterations}, openai, ,
"""

RUN_ENV_TEMPLATE = """RUN_NAME={run_name}
MODEL_PATH={model_path}
MODEL_ID={model_id}
PORT={port}
GPU={gpu}
VLLM_ARGS={vllm_args}
"""


def sanitize_run_id(run_name: str) -> str:
    return run_name.replace("-", "_").replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create isolated EQ-Bench run folders from a JSON spec.")
    parser.add_argument(
        "--spec",
        default="tools/model_runs.example.json",
        help="Path to JSON run spec list.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="EQ-Bench root directory.",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Relative folder under root where isolated run folders are created.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    spec_path = (root / args.spec).resolve() if not Path(args.spec).is_absolute() else Path(args.spec)

    with spec_path.open("r", encoding="utf-8") as f:
        specs = json.load(f)

    runs_root = root / args.runs_dir
    runs_root.mkdir(parents=True, exist_ok=True)

    root_template_path = root / "instruction-templates" / "OpenAIChat.yaml"
    if not root_template_path.exists():
        root_template_path.parent.mkdir(parents=True, exist_ok=True)
        root_template_path.write_text(OPENAI_CHAT_TEMPLATE, encoding="utf-8")

    for item in specs:
        run_name = item["run_name"]
        run_id = sanitize_run_id(run_name)
        model_path = item["model_path"]
        model_id = item.get("model_id", run_name)
        port = int(item.get("port", 8000))
        iterations = int(item.get("iterations", 10))
        gpu = str(item.get("gpu", "0"))
        vllm_args = item.get("vllm_args", "--dtype auto --gpu-memory-utilization 0.92")

        run_dir = runs_root / run_name
        (run_dir / "results").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "instruction-templates").mkdir(parents=True, exist_ok=True)

        config_text = CONFIG_TEMPLATE.format(
            port=port,
            run_id=run_id,
            model_id=model_id,
            iterations=iterations,
        )
        (run_dir / "config.cfg").write_text(config_text, encoding="utf-8")
        (run_dir / "instruction-templates" / "OpenAIChat.yaml").write_text(
            OPENAI_CHAT_TEMPLATE,
            encoding="utf-8",
        )

        run_env_text = RUN_ENV_TEMPLATE.format(
            run_name=run_name,
            model_path=model_path,
            model_id=model_id,
            port=port,
            gpu=gpu,
            vllm_args=vllm_args,
        )
        (run_dir / "run.env").write_text(run_env_text, encoding="utf-8")

        print(f"[OK] {run_dir}")

    print(f"\nCreated isolated run folders under: {runs_root}")


if __name__ == "__main__":
    main()
