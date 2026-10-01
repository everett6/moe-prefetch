"""
Will the expert cache engage on this GGUF, and how many slots fit?

The cache is only built for a layer whose routed experts are stored as SEPARATE
gate / up / down tensors with no per-expert scale or bias tensors
(llama-graph.cpp: `!gate_up_exps && gate_exps && down_exps && !*_b && !*_s`).
A GGUF converted with --fuse-gate-up-exps stores one fused `ffn_gate_up_exps`
tensor instead, and on that file the cache silently does nothing: the server
starts, prints no error, and runs every routed expert on the CPU. That is the
exact failure this project has paid for more than once -- a switch that did
not take effect looking like a switch that does not help -- so check the file
before spending a benchmark on it.

Reads the header only (tensor names, shapes, types); no weights are loaded.

    python3 experiments/gguf_preflight.py <model.gguf> [vram_budget_gib]
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/everett/llama.cpp-build/gguf-py")
from gguf import GGUFReader  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    budget_gib = float(sys.argv[2]) if len(sys.argv) > 2 else None
    r = GGUFReader(path, "r")

    def field(suffix):
        for k, f in r.fields.items():
            if k.endswith(suffix):
                try:
                    return f.contents()
                except Exception:
                    return None
        return None

    arch = field("general.architecture")
    print(f"file          {path}")
    print(f"architecture  {arch}   blocks {field('.block_count')}   "
          f"experts {field('.expert_count')}   routed per token {field('.expert_used_count')}")

    layers = {}
    total = 0
    for t in r.tensors:
        total += int(t.n_bytes)
        m = re.match(r"blk\.(\d+)\.(ffn_[a-z_]+_exps)(?:\.(\w+))?", t.name)
        if not m:
            continue
        il, kind, suffix = int(m.group(1)), m.group(2), m.group(3)
        layers.setdefault(il, {})[f"{kind}.{suffix}"] = t

    if not layers:
        print("\nNO routed-expert tensors found: this is not an MoE GGUF the cache can serve.")
        sys.exit(1)

    eligible, fused, extra = [], [], []
    expert_bytes = {}
    for il, ts in sorted(layers.items()):
        names = set(ts)
        if any(n.startswith("ffn_gate_up_exps") for n in names):
            fused.append(il)
            continue
        need = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}
        if not need <= names:
            extra.append((il, sorted(names)))
            continue
        others = names - need
        if others:
            extra.append((il, sorted(others)))
            continue
        n_expert = int(ts["ffn_up_exps.weight"].shape[-1])
        expert_bytes[il] = sum(int(ts[n].n_bytes) for n in need) / n_expert
        eligible.append(il)

    n_layers = len(layers)
    print(f"\nrouted-expert layers: {n_layers}")
    print(f"  cache-eligible (separate gate/up/down, no scale or bias): {len(eligible)}")
    if fused:
        print(f"  FUSED gate_up tensor: {len(fused)} layers -- the cache will NOT engage on "
              f"these (first: {fused[:4]})")
    if extra:
        print(f"  per-expert scale/bias or missing tensors: {len(extra)} layers -- NOT eligible "
              f"(e.g. layer {extra[0][0]}: {extra[0][1]})")

    if not eligible:
        print("\nVERDICT: the expert cache does nothing on this file. Re-convert without "
              "--fuse-gate-up-exps, or extend the cache chain to the fused layout first.")
        sys.exit(2)

    per = sorted(set(round(b) for b in expert_bytes.values()))
    mean_b = sum(expert_bytes.values()) / len(expert_bytes)
    routed_total = sum(int(t.n_bytes) for ts in layers.values() for t in ts.values())
    n_expert = int(layers[eligible[0]]["ffn_up_exps.weight"].shape[-1])
    print(f"\nexpert size   {mean_b/2**20:.2f} MiB mean "
          f"({len(per)} distinct sizes across layers: {[f'{b/2**20:.2f}' for b in per[:6]]})")
    print(f"routed experts {routed_total/2**30:.2f} GiB of {total/2**30:.2f} GiB; "
          f"everything else (must fit in VRAM with -ngl 999) {(total-routed_total)/2**30:.2f} GiB")
    print(f"one more slot on every layer costs {len(eligible)*mean_b/2**20:.0f} MiB")
    if len(per) > 1:
        print("  expert size varies by layer (mixed quant types): a slot is not the same "
              "price everywhere, so a per-layer slot profile should weigh bytes, not count.")

    if budget_gib:
        trunk = (total - routed_total) / 2**30
        room = budget_gib - trunk
        slots = int(room * 2**30 / (len(eligible) * mean_b))
        print(f"\nwith {budget_gib:g} GiB for model weights: trunk {trunk:.2f} GiB leaves "
              f"{room:.2f} GiB = about {slots} slots per layer "
              f"({100*slots/n_expert:.0f}% of {n_expert} experts resident)")
        print("  (context, compute buffers and the desktop need VRAM too -- leave ~1.5 GiB)")
    print(f"\nrun with:  -ngl 999 -ncmoe {field('.block_count')} --moe-expert-cache <slots> --no-mmap")
    if fused or extra:
        print("VERDICT: PARTIAL -- only the eligible layers are cached.")
        sys.exit(3)
    print("VERDICT: every routed-expert layer is cache-eligible.")


if __name__ == "__main__":
    main()
