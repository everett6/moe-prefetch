"""Shared settings and log parsing for the streaming tests (J4 to J7)."""
import re
from pathlib import Path

BASE = {"LLAMA_MOE_FUSED_CPU": "1", "LLAMA_MOE_SIDECAR": "1", "LLAMA_MOE_SIDECAR_TEAM": "1",
        "LLAMA_MOE_EARLY_ISSUE": "1", "LLAMA_MOE_BATCH_TABLES": "1"}
STREAM = dict(BASE, LLAMA_MOE_STREAM="1")

TOTALS = re.compile(r"moe-cache: stream (final|running) totals: steps=(\d+) offered=(\d+) landed=(\d+) "
                    r"in_time=(\d+) late=(\d+) unused=(\d+) gpu_miss=(\d+) failed=(\d)")
LEAD = re.compile(r"lead d1 ([\d.]+) us \((\d+)\), d2 ([\d.]+) us \((\d+)\)")


def counters(log: Path):
    """The last totals line of one server or runner log (final if present), and its lead times."""
    text = log.read_text(errors="replace")
    rows = TOTALS.findall(text)
    if not rows:
        return None
    final = [r for r in rows if r[0] == "final"]
    r = (final or rows)[-1]
    keys = ("steps", "offered", "landed", "in_time", "late", "unused", "gpu_miss", "failed")
    out = dict(zip(keys, map(int, r[1:])))
    leads = LEAD.findall(text)
    if leads:
        d1, n1, d2, n2 = leads[-1]
        out.update(lead_d1_us=float(d1), lead_d1_n=int(n1), lead_d2_us=float(d2), lead_d2_n=int(n2))
    return out


def summarize(rows):
    """Pooled J6 counters over several logs."""
    tot = {k: sum(r[k] for r in rows) for k in ("steps", "offered", "landed", "in_time", "late", "unused", "gpu_miss", "failed")}
    tot["served_from_vram"] = tot["in_time"] / max(tot["in_time"] + tot["gpu_miss"], 1)
    tot["copies_used"] = tot["in_time"] / max(tot["landed"], 1)
    n1 = sum(r.get("lead_d1_n", 0) for r in rows)
    n2 = sum(r.get("lead_d2_n", 0) for r in rows)
    tot["lead_d1_us"] = sum(r.get("lead_d1_us", 0) * r.get("lead_d1_n", 0) for r in rows) / max(n1, 1)
    tot["lead_d2_us"] = sum(r.get("lead_d2_us", 0) * r.get("lead_d2_n", 0) for r in rows) / max(n2, 1)
    return tot
