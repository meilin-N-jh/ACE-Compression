from __future__ import annotations

from typing import Any


def _normalize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def localized_record_fields(d: dict[str, Any], language: str) -> dict[str, Any]:
    if language == "en":
        return {
            "language": "en",
            "story": _normalize(d.get("STORY")),
            "question": _normalize(d.get("QUESTION")),
            "choices": {
                "A": _normalize(d.get("OPTION-A")),
                "B": _normalize(d.get("OPTION-B")),
                "C": _normalize(d.get("OPTION-C")),
                "D": _normalize(d.get("OPTION-D")),
            },
        }

    return {
        "language": "zh",
        "story": _normalize(d.get("故事")),
        "question": _normalize(d.get("问题")),
        "choices": {
            "A": _normalize(d.get("选项A")),
            "B": _normalize(d.get("选项B")),
            "C": _normalize(d.get("选项C")),
            "D": _normalize(d.get("选项D")),
        },
    }
