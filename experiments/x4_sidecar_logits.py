"""Sidecar exactness: every logit bit-identical to the fused CPU path under replayed placement.

RECORD runs the fused path on a live cache and records the placement map,
REPLAY-F replays it through the fused path (control: replay must be faithful),
REPLAY-S replays it through the sidecar. Gate: both replays have zero unequal
logits and routing traces identical to RECORD's.
"""
import sys

import x3_logits_replay as replay

ARMS = (("record", {"LLAMA_MOE_FUSED_CPU": "1"}, False),
        ("replay_f", {"LLAMA_MOE_FUSED_CPU": "1"}, True),
        ("replay_s", {"LLAMA_MOE_FUSED_CPU": "1", "LLAMA_MOE_SIDECAR": "1"}, True))

if __name__ == "__main__":
    replay.main(ARMS, "x4_logits")
    sys.exit(0)
