#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

class TagUpdate(BaseModel):
    tag_key: str = Field(description="Closed-set object tag from the decay table (e.g. person:fast, forklift:slow, cart:fast)")
    ttl_s: float = Field(ge=0.1, le=600.0, description="Recommended TTL in seconds")
    reason: Optional[str] = Field(default=None, description="Short explanation")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in this TTL update (0.0~1.0). High confidence (>=0.8) means strong evidence. Lower when proposing TTL above previously approved max.")


class GeminiProposal(BaseModel):
    updates: List[TagUpdate] = Field(description="List of TTL updates")
    notes: Optional[str] = Field(default=None, description="Any extra notes")


@dataclass
class EventCase:
    mission_id: str
    event_id: str
    tag_key: str
    detour_ratio: Optional[float] = None
    old_len: Optional[float] = None
    new_len: Optional[float] = None
    confidence: Optional[float] = None
    evidence: Optional[str] = None
    applied_ttl_s: Optional[float] = None
    timestamp: Optional[float] = None
    tag_group_id: Optional[str] = None
    repeat_count_in_mission: int = 1
    same_obstacle_reencountered: bool = False
    depth_ratio: Optional[float] = None
    depth_corrected_ttl_s: Optional[float] = None 
    vlm_dt_used_s: Optional[float] = None
    ttl_base_s: Optional[float] = None

    approval_mode: Optional[str] = None
    approval_status: Optional[str] = None
    proposed_ttl_s: Optional[float] = None
    approval_timestamp: Optional[float] = None

    human_feedback: Optional[str] = None
    human_feedback_norm: Optional[str] = None
    human_feedback_present: bool = False


METRIC_RE = re.compile(
    r"old\s*=\s*(?P<old>[0-9]+(?:\.[0-9]+)?)|"
    r"new\s*=\s*(?P<new>[0-9]+(?:\.[0-9]+)?)|"
    r"ratio\s*=\s*(?P<ratio>[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

def _normalize_feedback_text(text: Optional[str]) -> str:
    s = (text or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9_ ./:\-]", "", s)
    return s[:300]

def _read_key_file(path: str) -> Optional[str]:
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            s = f.read().strip()
            return s or None
    except Exception:
        return None


def _resolve_api_key(api_key_file: Optional[str] = None) -> Tuple[Optional[str], str]:
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key, "env"

    if api_key_file:
        key = _read_key_file(api_key_file)
        if key:
            return key, os.path.expanduser(api_key_file)

    candidates = [
        "~/.config/policy_bridge/gemini_api_key.txt",
        "~/.config/gemini_api_key.txt",
        "~/.gemini_api_key",
        "~/STeP_Cost/.secrets/gemini_api_key.txt",
    ]
    for c in candidates:
        key = _read_key_file(c)
        if key:
            return key, os.path.expanduser(c)

    return None, "default"


def _load_json(path: str, default: Any = None) -> Any:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        if default is None:
            raise FileNotFoundError(path)
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: Any) -> None:
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_tag_key(tag_key: str) -> str:
    s = (tag_key or "").strip().lower()
    s = s.replace("|", ":")
    s = re.sub(r"\s+", "", s)

    aliases = {
        "human": "person",
        "pedestrian": "person",
        "worker": "person",
        "staff": "person",
        "people": "person",
        "cartlike": "cart",
        "trolley": "cart",
        "mopcart": "cart",
        "truck": "forklift",
        "agv": "forklift",
        "vehicle": "forklift",
        "anomaly": "nav_anomaly",
        "obstacle": "nav_anomaly",
    }
    base_allowed = {"person", "forklift", "cart", "nav_anomaly", "workzone"}
    speed_suffixes = {"fast", "slow"}

    if ":" in s:
        base, suffix = s.split(":", 1)
        base = aliases.get(base, base) 
        if base in base_allowed and suffix in speed_suffixes:
            return f"{base}:{suffix}" 
        s = base 

    s = aliases.get(s, s)
    if s not in base_allowed:
        s = "nav_anomaly"
    return s


def split_tag(tag_key: str) -> str:
    return normalize_tag_key(tag_key)


def split_tag(tag_key: str) -> Tuple[str, str]:
    k = normalize_tag_key(tag_key)
    obj, mot = k.split(":", 1)
    return obj, mot


def extract_vlm_tag_key(vlm_field: Any) -> Optional[str]:
    if vlm_field is None:
        return None
    if isinstance(vlm_field, dict):
        tk = vlm_field.get("tag_key") or vlm_field.get("tag")
        return normalize_tag_key(str(tk)) if tk else None
    if isinstance(vlm_field, str):
        s = vlm_field.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                tk = obj.get("tag_key") or obj.get("tag")
                return normalize_tag_key(str(tk)) if tk else None
        except Exception:
            if ":" in s:
                return normalize_tag_key(s)
    return None


def parse_gemini_json(text: str) -> GeminiProposal:
    raw = (text or "").strip()
    try:
        return GeminiProposal.model_validate_json(raw)
    except Exception:
        pass

    if raw.startswith('"') and raw.endswith('"'):
        try:
            unquoted = json.loads(raw)
            if isinstance(unquoted, str):
                return GeminiProposal.model_validate_json(unquoted)
        except Exception:
            pass

    raw2 = re.sub(r"^```(?:json)?\s*", "", raw)
    raw2 = re.sub(r"\s*```$", "", raw2)
    m = re.search(r"\{[\s\S]*\}", raw2)
    if m:
        candidate = m.group(0)
        try:
            return GeminiProposal.model_validate_json(candidate)
        except Exception:
            pass

    obj = json.loads(raw2)
    return GeminiProposal.model_validate(obj)


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _extract_metrics_from_text(text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not text:
        return None, None, None
    old_len = new_len = ratio = None
    for m in METRIC_RE.finditer(text):
        gd = m.groupdict()
        if gd.get("old") is not None:
            old_len = _safe_float(gd["old"])
        if gd.get("new") is not None:
            new_len = _safe_float(gd["new"])
        if gd.get("ratio") is not None:
            ratio = _safe_float(gd["ratio"])
    return old_len, new_len, ratio


def _extract_event_case(ev: Dict[str, Any], mission_id: str, decay_table: Optional[Dict[str, Any]] = None) -> Optional[EventCase]:
    if not isinstance(ev, dict):
        return None
    raw_tag = ev.get("vlm_tag_key") or None
    if raw_tag:
        tag_key = normalize_tag_key(str(raw_tag))
    else:
        tag_key = extract_vlm_tag_key(ev.get("vlm"))
    if not tag_key:
        return None

    vlm = ev.get("vlm")
    conf = None
    evidence = None
    if isinstance(vlm, dict):
        conf = _safe_float(vlm.get("confidence"))
        evidence = vlm.get("evidence") or vlm.get("reason")
    elif isinstance(vlm, str):
        try:
            obj = json.loads(vlm)
            if isinstance(obj, dict):
                conf = _safe_float(obj.get("confidence"))
                evidence = obj.get("evidence") or obj.get("reason")
        except Exception:
            evidence = vlm[:300]

    trigger_text = ev.get("trigger_text") or ev.get("trigger") or ev.get("detour_trigger") or ""
    old_len, new_len, ratio = _extract_metrics_from_text(trigger_text)

    applied_ttl_s = None
    for key in ("applied_ttl_s", "ttl_s", "ttl"):
        if key in ev:
            applied_ttl_s = _safe_float(ev.get(key))
            break
    if applied_ttl_s is None and decay_table is not None:
        entry = decay_table.get(tag_key)
        if isinstance(entry, dict):
            applied_ttl_s = _safe_float(entry.get("ttl"))
        else:
            applied_ttl_s = _safe_float(entry)

    event_id = str(ev.get("event") or ev.get("event_id") or ev.get("id") or f"event_{int(time.time()*1000)}")
    ts = _safe_float(ev.get("timestamp") or ev.get("ts") or ev.get("time"))
    tag_group_id = ev.get("tag_group_id")
    repeat_count = ev.get("tag_repeat_count_in_mission", 1)
    try:
        repeat_count = int(repeat_count)
    except Exception:
        repeat_count = 1
    if repeat_count < 1:
        repeat_count = 1

    return EventCase(
        mission_id=str(mission_id or "unknown_mission"),
        event_id=event_id,
        tag_key=tag_key,
        detour_ratio=ratio,
        old_len=old_len,
        new_len=new_len,
        confidence=conf,
        evidence=evidence,
        applied_ttl_s=applied_ttl_s,
        timestamp=ts,
        tag_group_id=str(tag_group_id) if tag_group_id else None,
        repeat_count_in_mission=repeat_count,
        same_obstacle_reencountered=bool(repeat_count >= 2),
        depth_ratio=_safe_float(ev.get("depth_ratio")),
        depth_corrected_ttl_s=_safe_float(ev.get("depth_corrected_ttl_s")),
        vlm_dt_used_s=_safe_float(ev.get("vlm_dt_used_s")),
        ttl_base_s=_safe_float(ev.get("ttl_base_s")),
    )


def _extract_cases_from_mission_summary(ms: Dict[str, Any], decay_table: Optional[Dict[str, Any]] = None) -> List[EventCase]:
    mission_id = str(ms.get("mission_id") or ms.get("mission", {}).get("mission_id") or "unknown_mission")
    events = ms.get("events", []) or []
    out: List[EventCase] = []
    for ev in events:
        case = _extract_event_case(ev, mission_id, decay_table=decay_table)
        if case is not None:
            out.append(case)
    return out


def _load_archive_cases(archive_path: str) -> List[EventCase]:
    archive = _load_json(archive_path, default={})
    cases_raw = archive.get("cases", []) if isinstance(archive, dict) else []
    out: List[EventCase] = []
    for row in cases_raw:
        try:
            out.append(EventCase(**row))
        except Exception:
            continue
    return out


def _append_cases_to_archive(archive_path: str, cases: List[EventCase], max_cases: int = 2000) -> int:
    archive_path = os.path.expanduser(archive_path)
    archive = _load_json(archive_path, default={})
    if not isinstance(archive, dict):
        archive = {}
    old_rows = archive.get("cases", []) or []
    seen = {(row.get("mission_id"), row.get("event_id"), row.get("tag_key")) for row in old_rows if isinstance(row, dict)}
    added = 0
    for c in cases:
        key = (c.mission_id, c.event_id, c.tag_key)
        if key in seen:
            continue
        old_rows.append(asdict(c))
        seen.add(key)
        added += 1
    if len(old_rows) > max_cases:
        old_rows = old_rows[-max_cases:]
    archive["cases"] = old_rows
    archive["updated_at"] = time.time()
    _save_json(archive_path, archive)
    return added


def get_ttl_context_for_llm(
    archive_cases: List[EventCase],
    tag_keys: List[str],
    max_repeat1_cases: int = 30,
) -> Dict[str, Any]:
    from collections import defaultdict
    repeat_fail: Dict[str, List[float]] = defaultdict(list)
    repeat_ok:   Dict[str, List[EventCase]] = defaultdict(list)

    for c in archive_cases:
        status = c.approval_status or ""
        if status not in ("approved", "auto"):
            continue
        tag = normalize_tag_key(c.tag_key or "")
        if tag not in tag_keys:
            continue
        if c.proposed_ttl_s is None:
            continue

        if (c.repeat_count_in_mission or 1) >= 2:
            repeat_fail[tag].append(float(c.proposed_ttl_s))
        else:
            repeat_ok[tag].append(c)

    result: Dict[str, Any] = {}
    for tag in tag_keys:
        lower_bound = max(repeat_fail[tag]) if repeat_fail[tag] else None
        ok_cases = sorted(repeat_ok[tag], key=lambda x: x.timestamp or 0.0)
        ok_cases = ok_cases[-max_repeat1_cases:]

        result[tag] = {
            "lower_bound": lower_bound,
            "repeat_fail_n": len(repeat_fail[tag]),
            "repeat1_cases": ok_cases,
        }

    return result



def summarize_repeat_patterns(current_cases: List[EventCase]) -> Dict[str, Dict[str, Any]]:
    by_tag: Dict[str, Dict[str, Any]] = {}
    by_group: Dict[str, Dict[str, Any]] = {}
    for c in current_cases:
        tag_info = by_tag.setdefault(c.tag_key, {
            "events": 0,
            "max_repeat_count": 1,
            "reencounter_events": 0,
            "groups_repeated": 0,
        })
        tag_info["events"] += 1
        tag_info["max_repeat_count"] = max(tag_info["max_repeat_count"], int(c.repeat_count_in_mission or 1))
        if c.same_obstacle_reencountered:
            tag_info["reencounter_events"] += 1

        gid = c.tag_group_id or ""
        if gid:
            g = by_group.setdefault(gid, {
                "tag_key": c.tag_key,
                "event_ids": [],
                "max_repeat_count": 1,
            })
            g["event_ids"].append(c.event_id)
            g["max_repeat_count"] = max(g["max_repeat_count"], int(c.repeat_count_in_mission or 1))

    for gid, info in by_group.items():
        if info["max_repeat_count"] >= 2:
            by_tag.setdefault(info["tag_key"], {
                "events": 0,
                "max_repeat_count": 1,
                "reencounter_events": 0,
                "groups_repeated": 0,
            })["groups_repeated"] += 1

    return by_tag


def build_prompt(
    current_cases: List[EventCase],
    tag_counts: Dict[str, int],
    decay_table: Dict[str, Any],
    ttl_context: Dict[str, Any],
) -> str:
    lines: List[str] = []

    lines.append(
        "You are a TTL policy optimizer for a robot navigation system. "
        "applied_ttl = TTL_base x (1 - depth_ratio). "
        "Goal: find the MINIMUM TTL_base so the robot encounters each obstacle exactly once (repeat=1). "
        "Always propose ttl_base_s (not applied_ttl). Bias toward smaller values."
    )
    lines.append("")

    allowed_tags = sorted(tag_counts.keys())
    lines.append(
        f"IMPORTANT: You MUST use ONLY these exact tag_key values: {', '.join(allowed_tags)}. "
        "Do NOT invent or substitute other tag names (e.g. do not use 'vehicle' if 'forklift' is listed)."
    )
    lines.append("")

    lines.append("Rules:")
    lines.append("R1 repeat>=2: TTL_base too short -> RAISE, but incrementally (e.g. +10~20% per trial). Do NOT jump to a large value at once. Small consistent raises are preferred over a single large jump.")
    lines.append("R2 repeat=1, depth<0.5: applied_ttl was long -> HOLD or small decrease (not below confirmed_insufficient_bound).")
    lines.append("R3 repeat=1, depth>=0.5: applied_ttl was short -> RAISE proportionally (depth=0.8 -> only 20% applied -> large raise). Treat as weak signal.")
    lines.append("R4 Minimum: prefer lowest TTL_base with past repeat=1 and applied_ttl>=10s.")
    lines.append("R5 Converged (HOLD, confidence=0.9) if: confirmed_insufficient_bound exists AND current TTL_base > bound AND last 3+ repeat=1 cases within +-10% of current AND this mission repeat=1 with applied_ttl>=10s.")
    lines.append("confirmed_insufficient_bound: max TTL_base where repeat>=2 ever occurred. Do NOT propose at or below this. Confidence -0.25 if violated.")
    lines.append("")

    lines.append("Decay table (current):")
    for tag in sorted(tag_counts.keys()):
        entry = decay_table.get(tag)
        ttl = entry.get("ttl", "N/A") if isinstance(entry, dict) else (entry if entry is not None else "N/A")
        lines.append(f"  {tag}: {ttl}s")
    lines.append("")

    lines.append("Current mission:")
    repeat_summary = summarize_repeat_patterns(current_cases)
    for k in sorted(tag_counts.keys()):
        info = repeat_summary.get(k, {})
        lines.append(
            f"  {k}: count={tag_counts[k]} max_repeat={info.get('max_repeat_count',1)} "
            f"reencounters={info.get('reencounter_events',0)}"
        )
    for c in current_cases:
        dr  = f"{c.depth_ratio:.2f}" if c.depth_ratio is not None else "?"
        dct = f"{c.depth_corrected_ttl_s:.1f}s" if c.depth_corrected_ttl_s is not None else "?"
        base = f"{c.ttl_base_s:.1f}s" if c.ttl_base_s is not None else "?"
        lines.append(
            f"  tag={c.tag_key} repeat={c.repeat_count_in_mission} "
            f"ttl_base={base} depth={dr} applied={dct}"
        )
    lines.append("")

    lines.append("Past cases:")
    for tag in sorted(tag_counts.keys()):
        ctx           = ttl_context.get(tag, {})
        lower_bound   = ctx.get("lower_bound")
        repeat_fail_n = ctx.get("repeat_fail_n", 0)
        repeat1_cases = ctx.get("repeat1_cases", [])

        if lower_bound is not None:
            lines.append(f"  {tag}: confirmed_insufficient_bound={lower_bound:.1f}s (n={repeat_fail_n} failures)")
        else:
            lines.append(f"  {tag}: confirmed_insufficient_bound=unknown")

        if repeat1_cases:
            lines.append(f"  repeat=1 cases (oldest->newest, n={len(repeat1_cases)}):")
            for rc in repeat1_cases:
                p = f"{rc.proposed_ttl_s:.1f}s" if rc.proposed_ttl_s is not None else "?"
                d = f"{rc.depth_ratio:.2f}" if rc.depth_ratio is not None else "?"
                a = rc.applied_ttl_s
                a_str = f"{a:.1f}s" if a is not None else "?"
                weak = " [weak]" if (a is not None and a < 10.0) else ""
                lines.append(f"    ttl={p} depth={d} applied={a_str}{weak}")
        else:
            lines.append(f"  repeat=1 cases: none")
        lines.append("")

    lines.append(f'Output JSON only. Use ONLY these tag_key values: {", ".join(allowed_tags)}')
    lines.append('{"updates":[{"tag_key":"...","ttl_s":0.0,"confidence":0.0,"reason":"..."}],"notes":null}')

    return "\n".join(lines)


def apply_updates_to_decay_table(decay_table: Dict[str, Any], proposal: GeminiProposal) -> Tuple[Dict[str, Any], int]:
    updated = 0
    for u in proposal.updates:
        k = normalize_tag_key(u.tag_key)
        ttl = float(u.ttl_s)
        cur = decay_table.get(k)
        if isinstance(cur, dict):
            cur = dict(cur)
            cur["ttl"] = ttl
            decay_table[k] = cur
        else:
            decay_table[k] = {"ttl": ttl}
        updated += 1
    return decay_table, updated


def call_gemini(model: str, prompt: str, api_key_file: Optional[str] = None, max_retries: int = 6, retry_sleep_s: float = 1.0) -> GeminiProposal:
    api_key, key_src = _resolve_api_key(api_key_file)
    if not api_key:
        raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY/GOOGLE_API_KEY or provide --api_key_file or a default key file.")
    print(f"[INFO] Gemini API key source: {key_src}")
    client = genai.Client(api_key=api_key)

    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GeminiProposal,
        temperature=0.2,
        top_p=0.9,
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            proposal = parse_gemini_json(resp.text)
            for upd in proposal.updates:
                upd.tag_key = normalize_tag_key(upd.tag_key)
            return proposal
        except Exception as e:
            last_err = e
            print(f"[WARN] Gemini call failed attempt={attempt}/{max_retries}: {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(retry_sleep_s * attempt)
                print(f"[WARN] Retrying Gemini call attempt={attempt+1}/{max_retries} ...")

    raise RuntimeError(f"Gemini call failed after retries: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--decay_table_path", required=True)
    ap.add_argument("--init_table_if_missing", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Print proposal but do not write decay_table")
    ap.add_argument("--update_mission_summary", default=None)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--api_key_file", default=None)
    ap.add_argument("--retrieval_archive_path", default="~/.ros/llm_decay_rag_archive.json")
    ap.add_argument("--retrieval_max_repeat1_cases", type=int, default=30)
    ap.add_argument("--append_to_archive", action="store_true")
    ap.add_argument("--archive_max_cases", type=int, default=2000)
    ap.add_argument("--approve_apply", action="store_true") 
    args = ap.parse_args()

    ms = _load_json(args.input, default={})
    mission_id = ms.get("mission_id") or ms.get("mission", {}).get("mission_id") or "unknown_mission"

    if (not os.path.exists(os.path.expanduser(args.decay_table_path))) and args.init_table_if_missing:
        _save_json(args.decay_table_path, {})
    decay_table = _load_json(args.decay_table_path, default={})

    current_cases = _extract_cases_from_mission_summary(ms, decay_table=decay_table)
    tag_counts: Dict[str, int] = {}
    for c in current_cases:
        tag_counts[c.tag_key] = tag_counts.get(c.tag_key, 0) + 1

    print(f"[INFO] mission_id={mission_id} events_total={len(ms.get('events', []) or [])} events_with_vlm={len(current_cases)} unique_tags={len(tag_counts)}")
    print(f"[INFO] decay_table tags={len(decay_table)} path={args.decay_table_path}")

    if not tag_counts:
        print("[INFO] No VLM-tagged events found. Nothing to update.")
        _save_json(args.output, ms)
        return 0

    archive_cases = _load_archive_cases(args.retrieval_archive_path)
    archive_cases = [c for c in archive_cases if c.mission_id != str(mission_id)]

    ttl_context = get_ttl_context_for_llm(
        archive_cases,
        list(tag_counts.keys()),
        max_repeat1_cases=args.retrieval_max_repeat1_cases,
    )

    total_repeat1 = sum(len(ctx["repeat1_cases"]) for ctx in ttl_context.values())
    total_fail    = sum(ctx["repeat_fail_n"]       for ctx in ttl_context.values())
    print(
        f"[INFO] RAG archive={os.path.expanduser(args.retrieval_archive_path)} "
        f"archive_cases={len(archive_cases)} "
        f"repeat1_cases_loaded={total_repeat1} "
        f"repeat_fail_cases={total_fail} "
        f"max_repeat1_cases={args.retrieval_max_repeat1_cases}"
    )

    prompt = build_prompt(current_cases, tag_counts, decay_table, ttl_context)

    print(f"[INFO] Calling Gemini model={args.model} (structured output via response_schema) ...")
    proposal = call_gemini(model=args.model, prompt=prompt, api_key_file=args.api_key_file)

    filtered_updates = []
    for u in proposal.updates:
        k = normalize_tag_key(u.tag_key)
        if k in tag_counts:
            filtered_updates.append(u)
    proposal.updates = filtered_updates

    if not proposal.updates:
        print("[INFO] Gemini returned no applicable updates (after filtering).")
        _save_json(args.output, ms)
        return 0

    print("[PROPOSAL] TTL updates:")
    for u in proposal.updates:
        print(f"  - {u.tag_key}: ttl_s={u.ttl_s:.2f}" + (f"  # {u.reason}" if u.reason else ""))
    import json as _json
    print("[PROPOSAL_JSON] " + _json.dumps({
        "updates": [
            {"tag_key": u.tag_key, "ttl_s": u.ttl_s, "reason": u.reason, "confidence": getattr(u, "confidence", 0.5)}
            for u in proposal.updates
        ]
    }, ensure_ascii=False))

    if not args.yes:
        ans = input("Apply these updates to decay_table.json? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("[INFO] Not applied.")
            _save_json(args.output, ms)
            return 0

    if args.dry_run:
        print("[INFO] dry_run=True: proposal generated but decay_table NOT modified")
        _save_json(args.output, ms)
        return 0

    decay_table, n = apply_updates_to_decay_table(decay_table, proposal)
    _save_json(args.decay_table_path, decay_table)
    print(f"[INFO] Applied updates={n} -> saved decay_table: {args.decay_table_path}")

    ms_out = dict(ms)
    ms_out.setdefault("llm_policy_update", {})
    ms_out["llm_policy_update"] = {
        "model": args.model,
        "updated_tags": [u.tag_key for u in proposal.updates],
        "notes": proposal.notes,
        "timestamp": time.time(),
        "rag": {
            "archive_path": os.path.expanduser(args.retrieval_archive_path),
            "archive_cases_considered": len(archive_cases),
            "ttl_context": {
                tag: {
                    "lower_bound": ctx.get("lower_bound"),
                    "repeat_fail_n": ctx.get("repeat_fail_n", 0),
                    "repeat1_cases_n": len(ctx.get("repeat1_cases", [])),
                }
                for tag, ctx in ttl_context.items()
            },
        },
    }

    _save_json(args.output, ms_out)
    print(f"[INFO] Wrote output: {args.output}")

    if args.update_mission_summary:
        _save_json(args.update_mission_summary, ms_out)
        print(f"[INFO] Updated mission_summary in-place: {args.update_mission_summary}")

    if args.append_to_archive:
        proposal_map = {u.tag_key: float(u.ttl_s) for u in proposal.updates}
        for c in current_cases:
            if c.tag_key in proposal_map:
                c.proposed_ttl_s = proposal_map[c.tag_key]
            c.approval_mode = "auto"
            c.approval_status = "auto"
            c.approval_timestamp = time.time()
        added = _append_cases_to_archive(args.retrieval_archive_path, current_cases, max_cases=args.archive_max_cases)
        print(f"[INFO] Appended current cases to RAG archive: added={added} path={os.path.expanduser(args.retrieval_archive_path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
