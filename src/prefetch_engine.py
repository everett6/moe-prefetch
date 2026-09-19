"""
The prefetch engine: a GPU-resident expert cache fed by real async H2D copies.

This is the runtime piece, written against PyTorch so it can be built and
measured today. The eventual home for this logic is llama.cpp's MoE graph, which
needs the CUDA toolkit installed as root; the point of building it here first is
that everything except the graph surgery can be validated without that -- cache
behaviour, eviction, scheduling, and above all whether the copies actually land
in time when a real GPU is doing real work on another stream.

Design, and why each piece is the way it is:

  cache        one fixed [capacity, bytes_per_expert] uint8 tensor per layer,
               allocated once. Experts are copied into slots; a dict maps
               expert id -> slot. Fixed slots rather than dynamic allocation
               because the C++ side will have the same constraint, and because
               allocator churn would dominate the measurement here.
  eviction     LRU. AI2's simulation put LRU at 85.4% expert hit on real traces
               at capacity 66, and it is what llama.cpp PR #27861 uses, so it is
               the honest baseline to improve on rather than a strawman.
  copies       issued on a DEDICATED CUDA STREAM from PINNED host memory with
               non_blocking=True. All three are required for overlap: a pageable
               source forces a synchronising staging copy, and the default
               stream would serialise against compute.
  deadline     each prefetch records a CUDA event. `wait_ready` checks whether
               that event has completed by the time the layer actually needs the
               experts -- which is the whole question. A prefetch that arrives
               late is not a hit, and counting it as one would be the easiest way
               to produce a flattering number here.

The engine does not hold real expert weights. Sizes are real (2.92 MiB per
expert, read from the Q4_K_M GGUF in AI2's PREFETCH_FEASIBILITY.md) and the
bytes move for real, but their content is irrelevant to transfer timing, so one
host-side staging buffer is reused across layers. That keeps host memory at
~374 MB instead of the 17.9 GB the whole expert set would need.
"""
from collections import OrderedDict

import torch

EXPERT_MIB = 2.92                       # measured from the Q4_K_M GGUF tensor table
EXPERT_BYTES = int(EXPERT_MIB * 1024 * 1024)


class ExpertCache:
    """Per-layer LRU cache of experts resident in VRAM, with async refill."""

    def __init__(self, n_layers, n_experts, capacity, device="cuda",
                 bytes_per_expert=EXPERT_BYTES):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.capacity = capacity
        self.device = device
        self.bytes_per_expert = bytes_per_expert

        # One slab per layer. uint8 because we are moving bytes, not maths.
        self.slabs = [torch.empty((capacity, bytes_per_expert), dtype=torch.uint8,
                                  device=device) for _ in range(n_layers)]
        # expert id -> slot index, in LRU order (front = least recently used)
        self.maps = [OrderedDict() for _ in range(n_layers)]
        self.free = [list(range(capacity)) for _ in range(n_layers)]

        # Host staging, pinned. Content is irrelevant to timing; size is not.
        self.staging = torch.empty((n_experts, bytes_per_expert), dtype=torch.uint8,
                                   pin_memory=True)

        self.copy_stream = torch.cuda.Stream(device=device)
        self.pending = {}                # (layer, expert) -> cuda event
        self.stats = {"hits": 0, "misses": 0, "late": 0, "fetches": 0,
                      "bytes": 0, "evictions": 0}

    # -- internals ---------------------------------------------------------
    def _slot_for(self, layer, expert):
        """Reserve a slot for `expert` in `layer`, evicting LRU if needed."""
        m = self.maps[layer]
        if expert in m:
            m.move_to_end(expert)
            return m[expert], False
        if self.free[layer]:
            slot = self.free[layer].pop()
        else:
            old, slot = m.popitem(last=False)     # least recently used
            self.pending.pop((layer, old), None)
            self.stats["evictions"] += 1
        m[expert] = slot
        return slot, True

    # -- public ------------------------------------------------------------
    def prefetch(self, layer, expert_ids):
        """Start async copies for `expert_ids` into layer's cache. Returns count."""
        started = 0
        with torch.cuda.stream(self.copy_stream):
            for e in expert_ids:
                e = int(e)
                if e in self.maps[layer]:
                    self.maps[layer].move_to_end(e)
                    continue
                slot, _ = self._slot_for(layer, e)
                self.slabs[layer][slot].copy_(self.staging[e], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.copy_stream)
                self.pending[(layer, e)] = ev
                started += 1
                self.stats["fetches"] += 1
                self.stats["bytes"] += self.bytes_per_expert
        return started

    def query(self, layer, expert_ids):
        """Which of `expert_ids` are usable NOW, without waiting?

        An expert whose copy is still in flight is NOT a hit: the layer would
        have to block on it. Those are counted separately as `late`, because the
        difference between "predicted correctly" and "predicted correctly and
        arrived in time" is the entire question this engine exists to answer.
        """
        hit, late, miss = [], [], []
        for e in expert_ids:
            e = int(e)
            if e not in self.maps[layer]:
                miss.append(e)
                continue
            ev = self.pending.get((layer, e))
            if ev is not None and not ev.query():
                late.append(e)
            else:
                self.pending.pop((layer, e), None)
                self.maps[layer].move_to_end(e)
                hit.append(e)
        self.stats["hits"] += len(hit)
        self.stats["misses"] += len(miss)
        self.stats["late"] += len(late)
        return hit, late, miss

    def admit(self, layer, expert_ids):
        """A miss was computed on the CPU; bring it in so the next token can use it.

        This MUST move the bytes. An earlier version marked the expert resident
        without copying, which gave the no-prefetch LRU policy free hits and made
        it report 0.0 GB of traffic -- flattering exactly the baseline the
        predictor is being compared against.
        """
        with torch.cuda.stream(self.copy_stream):
            for e in expert_ids:
                e = int(e)
                if e in self.maps[layer]:
                    self.maps[layer].move_to_end(e)
                    continue
                slot, _ = self._slot_for(layer, e)
                self.slabs[layer][slot].copy_(self.staging[e], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.copy_stream)
                self.pending[(layer, e)] = ev
                self.stats["fetches"] += 1
                self.stats["bytes"] += self.bytes_per_expert

    def sync(self):
        self.copy_stream.synchronize()

    def vram_bytes(self):
        return self.n_layers * self.capacity * self.bytes_per_expert

    def summary(self):
        s = dict(self.stats)
        total = s["hits"] + s["misses"] + s["late"]
        s["expert_hit_rate"] = s["hits"] / max(total, 1)
        s["late_rate"] = s["late"] / max(total, 1)
        s["gb_moved"] = s["bytes"] / 1e9
        return s
