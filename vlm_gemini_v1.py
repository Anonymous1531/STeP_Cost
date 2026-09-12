#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field


class VLMResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tag_key: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(pattern=r"^[a-z0-9_]{1,64}$")


_NON_TAG_KEYS = {"ttl", "lambda", "updated_at", "cases", "version"}
_DEFAULT_BASE_TAGS = ["person", "forklift", "cart"]


def _normalize_base_tag(value: Any) -> str:
    s = str(value or "").strip().lower().replace("|", ":")
    s = re.sub(r"\s+", "", s)
    if ":" in s:
        s = s.split(":", 1)[0]
    aliases = {
        "human": "person",
        "pedestrian": "person",
        "worker": "person",
        "people": "person",
        "fork": "forklift",
        "forklifttruck": "forklift",
        "lifttruck": "forklift",
        "trolley": "cart",
        "dolly": "cart",
        "handtruck": "cart",
        "shoppingcart": "cart",
    }
    return aliases.get(s, s)


def load_allowed_base_tags(decay_table_path: str) -> List[str]:
    path = Path(os.path.expanduser(decay_table_path))
    if not path.exists():
        return list(_DEFAULT_BASE_TAGS)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return list(_DEFAULT_BASE_TAGS)
    if not isinstance(raw, dict):
        return list(_DEFAULT_BASE_TAGS)

    tags = set()
    for key in raw.keys():
        if not isinstance(key, str) or key.lower() in _NON_TAG_KEYS:
            continue
        base = _normalize_base_tag(key)
        if base:
            tags.add(base)
    return sorted(tags) if tags else list(_DEFAULT_BASE_TAGS)


def _read_key_file(path: str) -> Optional[str]:
    try:
        value = Path(os.path.expanduser(path)).read_text(encoding="utf-8").strip()
        return value or None
    except Exception:
        return None


def load_api_key(api_key_file: str = "") -> Optional[str]:
    value = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if value:
        return value.strip()
    if api_key_file:
        value = _read_key_file(api_key_file)
        if value:
            return value
    for candidate in (
        "~/.config/policy_bridge/gemini_api_key.txt",
        "~/.config/gemini_api_key.txt",
        "~/.gemini_api_key",
        "~/STeP_Cost/.secrets/gemini_api_key.txt",
    ):
        value = _read_key_file(candidate)
        if value:
            return value
    return None


def read_image_bytes(path: str, max_dim: int = 640, jpeg_quality: int = 80) -> bytes:
    try:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        w, h = image.size
        scale = min(1.0, float(max_dim) / max(w, h))
        if scale < 1.0:
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=jpeg_quality)
        return buf.getvalue()
    except Exception:
        return Path(path).read_bytes()


def build_prompt(user_prompt: str, allowed_tags: List[str]) -> str:
    tag_list = ", ".join(allowed_tags)
    return (
        f"{user_prompt.strip()}\n\n"
        f"ALLOWED tag_key values (use ONLY one of these exactly): {tag_list}\n"
        "Identify the OBJECT TYPE only (do NOT include speed class).\n"
        "Do NOT append :slow or :fast. Return base tag only.\n"
        "IMPORTANT: You MUST choose one of the allowed tags.\n"
        "If the object is ambiguous, choose the closest allowed tag, assign a low confidence score, "
        "and describe the ambiguity in evidence.\n"
        "STRICT OUTPUT FORMAT:\n"
        "- Return exactly one JSON object.\n"
        '- Example: {"tag_key":"person","confidence":0.92,'
        '"evidence":"person_walking_center_aisle"}\n'
        "- No markdown, no code fences, no explanation.\n"
        "- Keys must be exactly: tag_key, confidence, evidence.\n"
        "- Evidence: lowercase letters, numbers, underscores only. Max 64 chars.\n"
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty Gemini response")
    candidates = [raw]
    if raw.startswith('"') and raw.endswith('"'):
        try:
            unquoted = json.loads(raw)
            if isinstance(unquoted, str):
                candidates.append(unquoted.strip())
        except Exception:
            pass
    stripped = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    candidates.append(stripped)
    m = re.search(r"\{[\s\S]*\}", stripped)
    if m:
        candidates.append(m.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    raise ValueError("could not parse Gemini JSON output")


def _validate_result(obj: Dict[str, Any], allowed_tags: List[str]) -> VLMResult:
    result = VLMResult.model_validate(obj)
    tag = _normalize_base_tag(result.tag_key)
    if tag not in allowed_tags:
        raise ValueError(f"tag_key {tag!r} not in allowed tags {allowed_tags}")
    evidence = result.evidence.strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{1,64}", evidence):
        raise ValueError("invalid evidence format")
    return VLMResult(tag_key=tag, confidence=float(result.confidence), evidence=evidence)


def call_gemini_vlm(
    model: str,
    prompt: str,
    image_paths: List[str],
    api_key_file: str,
) -> VLMResult:
    api_key = load_api_key(api_key_file)
    if not api_key:
        raise RuntimeError("Gemini API key not found")

    client = genai.Client(api_key=api_key)
    parts = [types.Part.from_text(text=prompt)]
    for path in image_paths:
        parts.append(types.Part.from_bytes(data=read_image_bytes(path), mime_type="image/jpeg"))

    kwargs: Dict[str, Any] = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_schema": VLMResult,
        "max_output_tokens": 512,
    }
    try:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass

    response = client.models.generate_content(
        model=model[7:] if model.startswith("models/") else model,
        contents=parts,
        config=types.GenerateContentConfig(**kwargs),
    )

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, VLMResult):
        return parsed
    if isinstance(parsed, dict):
        return VLMResult.model_validate(parsed)
    return VLMResult.model_validate(_extract_json_object(getattr(response, "text", "") or ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--decay_table_path", default="")
    ap.add_argument("--api_key_file", default="")
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    allowed_tags = load_allowed_base_tags(args.decay_table_path)
    prompt = build_prompt(args.prompt, allowed_tags)

    last_error: Optional[Exception] = None
    for attempt in range(max(1, args.retries + 1)):
        try:
            raw_result = call_gemini_vlm(
                model=args.model,
                prompt=prompt,
                image_paths=args.images,
                api_key_file=args.api_key_file,
            )
            result = _validate_result(raw_result.model_dump(), allowed_tags)
            print(json.dumps(result.model_dump(), ensure_ascii=False))
            return 0
        except Exception as e:
            last_error = e
            if attempt < args.retries:
                time.sleep(0.5 * (attempt + 1))

    print(
        json.dumps({"error": f"{type(last_error).__name__}: {last_error}"}, ensure_ascii=False),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
