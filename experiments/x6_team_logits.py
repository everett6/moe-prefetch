"""J1: the persistent team computes exactly what one graph per miss computes.

RECORD runs the sidecar on a live cache and records the placement map,
REPLAY-S replays it through the sidecar (control), REPLAY-T through the
sidecar with the persistent team. Gate: both replays have zero unequal logits
and routing traces identical to RECORD's. Pre-registered in docs/STREAMING-IMPLEMENTATION.md.
"""
import sys

import x3_logits_replay as replay

SIDECAR = {"LLAMA_MOE_FUSED_CPU": "1", "LLAMA_MOE_SIDECAR": "1"}
ARMS = (("record", SIDECAR, False),
        ("replay_s", SIDECAR, True),
        ("replay_t", dict(SIDECAR, LLAMA_MOE_SIDECAR_TEAM="1"), True))

if __name__ == "__main__":
    replay.main(ARMS, "x6_team_logits")
    sys.exit(0)
