#!/usr/bin/env python3
import argparse, json, os, sys, time, re
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types

_NON_TAG_KEYS = {"ttl", "lambda", "updated_at", "cases", "version"}

def load_allowed_tags(decay_table_path: str) -> List[str]:
    if not decay_table_path or not os.path.exists(decay_table_path):
        return []
    try:
        d = json.load(open(decay_table_path, "r", encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(d, dict):
        return []

    tags = set()
    for k in d.keys():
        if not isinstance(k, str) or len(k) > 80:
            continue
        if k.lower() in _NON_TAG_KEYS:
            continue
        nk = _normalize_tag_key(k)
        if nk:
            tags.add(nk)
    return sorted(tags)


def build_schema(allowed_tags: List[str]) -> Dict[str, Any]:
    tag_schema: Dict[str, Any] = {"type": "string"}
    if allowed_tags:
        tag_schema["enum"] = allowed_tags
    else:
        tag_schema["description"] = "tag_key from the allowed closed set"

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["tag_key", "confidence", "evidence"],
        "properties": {
            "tag_key": tag_schema,
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "evidence": {"type": "string",
                "pattern": "^[a-z0-9_]{1,64}$",
                "description": "short snake_case evidence, max 64 chars"},
        },
    }


def read_image_bytes(path: str, max_dim: int = 640, jpeg_quality: int = 80) -> bytes:
    try:
        from PIL import Image
        import io
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(1.0, float(max_dim) / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        return buf.getvalue()
    except Exception:
        with open(path, "rb") as f:
            return f.read()


def _read_api_key_file(path: str) -> Optional[str]:
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            key = f.read().strip()
        return key or None
    except Exception:
        return None


def load_api_key(api_key_file: str = "") -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key.strip()

    if api_key_file:
        key = _read_api_key_file(api_key_file)
        if key:
            return key

    candidates = [
        "~/.config/policy_bridge/gemini_api_key.txt",
        "~/.config/gemini_api_key.txt",
        "~/.gemini_api_key",
        "~/STeP_Cost/.secrets/gemini_api_key.txt",
    ]
    for p in candidates:
        key = _read_api_key_file(p)
        if key:
            return key

    return None


def call_gemini_vlm(
    model: str,
    prompt: str,
    image_paths: List[str],
    schema: Dict[str, Any], 
    api_key_file: str = "",
) -> dict:
    if model.startswith("models/"):
        model = model[len("models/"):]

    api_key = load_api_key(api_key_file)
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Put it in ~/.config/policy_bridge/gemini_api_key.txt "
            "or pass --api_key_file, or set GEMINI_API_KEY/GOOGLE_API_KEY."
        )

    client = genai.Client(api_key=api_key)

    parts = [types.Part.from_text(text=prompt)]
    for p in image_paths:
        b = read_image_bytes(p)
        parts.append(types.Part.from_bytes(data=b, mime_type="image/jpeg"))

    thinking_cfg = None
    try:
        thinking_cfg = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass

    cfg_kwargs: Dict[str, Any] = dict(
        temperature=0,
        response_mime_type="application/json",
        max_output_tokens=1024,
    )
    if thinking_cfg is not None:
        cfg_kwargs["thinking_config"] = thinking_cfg

    cfg = types.GenerateContentConfig(**cfg_kwargs)

    resp = client.models.generate_content(model=model, contents=parts, config=cfg)

    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, dict):
        return parsed

    txt = (resp.text or "").strip()
    if txt:
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        try:
            candidate = _extract_braced(txt)
            if candidate and candidate != txt:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
        except Exception:
            pass

    parse_errors = []
    try:
        for cand in (resp.candidates or []):
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in (getattr(content, "parts", None) or []):
                t = getattr(part, "text", None)
                s = str(t).strip() if t is not None else ""
                if not s:
                    continue

                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        return obj
                except Exception as e:
                    parse_errors.append(f"part_direct={type(e).__name__}:{s[:120]!r}")

                try:
                    candidate = _extract_braced(s)
                    if candidate:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                except Exception as e:
                    parse_errors.append(f"part_braced={type(e).__name__}:{s[:120]!r}")
    except Exception as e:
        parse_errors.append(f"candidate_walk={type(e).__name__}:{e}")

    finish_reasons = []
    for cand in (getattr(resp, "candidates", None) or []):
        fr = str(getattr(cand, "finish_reason", None))
        finish_reasons.append(fr)

    debug = {
        "resp_text_head": txt[:160],
        "resp_text_len": len(txt),
        "has_parsed": getattr(resp, "parsed", None) is not None,
        "candidate_count": len(getattr(resp, "candidates", None) or []),
        "finish_reasons": finish_reasons,
        "parse_errors": parse_errors[:4],
        "candidate_debug": [],
    }

    if any("MAX_TOKENS" in r or "max_tokens" in r.lower() for r in finish_reasons):
        debug["truncation_warning"] = (
            "Response was truncated by max_output_tokens. "
            "Increase max_output_tokens further if this persists."
        )

    for cand in (getattr(resp, "candidates", None) or []):
        cand_info = {
            "finish_reason": str(getattr(cand, "finish_reason", None)),
            "token_count": getattr(cand, "token_count", None),
        }
        content = getattr(cand, "content", None)
        if content:
            cand_info["parts"] = []
            for part in (getattr(content, "parts", None) or []):
                cand_info["parts"].append({
                    "has_text": bool(getattr(part, "text", None)),
                    "text_head": str(getattr(part, "text", "") or "")[:80],
                })
        debug["candidate_debug"].append(cand_info)

    raise ValueError(f"Gemini non-JSON response debug={json.dumps(debug, ensure_ascii=False)}")

_TAG_RE = re.compile(r'"?tag_key"?\s*:\s*"([^"]+)"', re.IGNORECASE)
_CONF_RE = re.compile(r'"?confidence"?\s*:\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)


def _extract_braced(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    i = t.find("{")
    j = t.rfind("}")
    return t[i:j + 1] if (i >= 0 and j > i) else t


def _repair_json_loose(text: str) -> str:
    t = _extract_braced(text)
    t = re.sub(r'(?<=\{|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'"\1":', t)
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return t

def _normalize_tag_key(tag_key: str) -> str:
    s = (tag_key or "").strip().lower()
    s = s.replace("|", ":")
    s = re.sub(r"\s+", "", s)

    if ":" in s:
        s = s.split(":", 1)[0]

    base_aliases = {
        "human": "person",
        "pedestrian": "person",
        "worker": "person",
        "staff": "person",
        "operator": "person",
        "people": "person",
        "fork": "forklift",
        "forklifttruck": "forklift",
        "lifttruck": "forklift",
        "mop": "mopcart",
        "cleaningcart": "mopcart",
        "janitorcart": "mopcart",
        "cartlike": "cart",
        "trolley": "cart",
        "dolly": "cart",
        "handtruck": "cart",
        "shoppingcart": "cart",
        "agv": "delivery_robot",
        "amr": "delivery_robot",
        "deliveryrobot": "delivery_robot",
        "autonomousmobilerobot": "delivery_robot",
        "mobilerobot": "delivery_robot",
        "robot": "delivery_robot",
        "anomaly": "nav_anomaly",
        "obstacle": "nav_anomaly",
        "debris": "nav_anomaly",
        "spill": "nav_anomaly",
        "unknown": "nav_anomaly",
    }
    return base_aliases.get(s, s)


def _parse_vlm_json(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty Gemini response")

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    if raw.startswith('"') and raw.endswith('"'):
        try:
            unquoted = json.loads(raw)
            if isinstance(unquoted, str):
                obj = json.loads(unquoted)
                if isinstance(obj, dict):
                    return obj
        except Exception:
            pass

    raw2 = re.sub(r"^```(?:json)?\s*", "", raw)
    raw2 = re.sub(r"\s*```$", "", raw2)
    m = re.search(r"\{[\s\S]*\}", raw2)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    repaired = _repair_json_loose(raw2)
    obj = json.loads(repaired)
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object")
    return obj

def _normalize_and_validate_evidence(evidence_raw: str) -> str:
    e = (evidence_raw or "").strip().lower()
    if not e:
        raise ValueError("Missing evidence")

    if not re.fullmatch(r"[a-z0-9_]{1,64}", e):
        raise ValueError(f"Invalid evidence format: {e!r}")

    if len([tok for tok in e.split("_") if tok]) > 8:
        raise ValueError(f"Evidence too long: {e!r}")

    return e

def _validate_vlm_obj(obj: dict, allowed_tags: list[str]) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("VLM output is not a JSON object")

    tag = obj.get("tag_key")
    if tag is None:
        tag = obj.get("tag")
    tag = _normalize_tag_key(str(tag or ""))
    if not tag:
        raise ValueError("Missing tag_key")

    if allowed_tags and tag not in allowed_tags:
        fallback = allowed_tags[0]
        import sys
        print(f"[WARN] tag_key '{tag}' not in allowed_tags {allowed_tags}, fallback to '{fallback}'", file=sys.stderr)
        tag = fallback
        obj["tag_key"] = tag
        obj["confidence"] = min(float(obj.get("confidence", 0.5)), 0.5)  

    conf_raw = obj.get("confidence")
    if conf_raw is None:
        raise ValueError("Missing confidence")
    try:
        conf = float(conf_raw)
    except Exception:
        raise ValueError(f"Invalid confidence: {conf_raw!r}")

    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"confidence out of range: {conf}")

    evidence = obj.get("evidence", "")
    if evidence is None:
        evidence = ""
    evidence = _normalize_and_validate_evidence(str(evidence))

    return {
        "tag_key": tag,
        "confidence": conf,
        "evidence": evidence,
    }

def main():
    print("VLM_SCRIPT_MARKER_20260324", file=sys.stderr, flush=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--decay_table_path", default="")
    ap.add_argument("--api_key_file", default="")
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    allowed_tags = load_allowed_tags(args.decay_table_path)
    schema = build_schema(allowed_tags)

    last_err = None

    for i in range(args.retries + 1):
        try:
            tag_list_str = (
                ", ".join(allowed_tags) if allowed_tags
                else "person, forklift, cart, nav_anomaly"
            )
            prompt = (
                args.prompt
                + f"\n\nALLOWED tag_key values (use ONLY one of these exactly): {tag_list_str}\n"
                + "Identify the OBJECT TYPE only (do NOT include speed class):\n"
                + "  person: human, worker, pedestrian, anyone walking\n"
                + "  forklift: lift truck, fork truck, pallet jack, industrial vehicle with forks\n"
                + "  cart: shopping cart, transport cart, trolley, dolly, hand cart, wheeled cart, push cart\n"
                + "Do NOT append :slow or :fast. Return base tag only.\n"
                + "IMPORTANT: You MUST choose one of the allowed tags. Do NOT use nav_anomaly.\n"
                + "If unsure, pick the visually closest tag among person, forklift, cart.\n"
                + "\nSTRICT OUTPUT FORMAT:\n"
                "- Return exactly one JSON object.\n"
                '- Example: {"tag_key":"person","confidence":0.92,"evidence":"person_walking_center_aisle"}\n'
                "- No markdown, no code fences, no explanation.\n"
                "- Keys must be exactly: tag_key, confidence, evidence.\n"
                "- Evidence: lowercase letters, numbers, underscores only. Max 64 chars.\n"
            )
            if i > 0:
                prompt += (
                    f"\nRetry #{i}: previous response had an invalid tag_key or format. "
                    f"You MUST use one of: {tag_list_str} (NO speed class suffix)"
                )

            obj = call_gemini_vlm(
                args.model,
                prompt,
                args.images,
                schema,
                args.api_key_file,
            )

            obj = _validate_vlm_obj(obj, allowed_tags)
            print(json.dumps(obj, ensure_ascii=False))
            return 0

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(0.5 * (i + 1))

    print(json.dumps({
        "error": last_err,
    }, ensure_ascii=False), file=sys.stderr)
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
