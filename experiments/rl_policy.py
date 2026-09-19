"""
Stage D: reward-driven optimisation of the prefetch decision.

## The learning problem, stated properly

Calling something "RL" without defining the MDP is how this kind of stage becomes
decoration, so:

STATE s  -- exactly what the host has at layer L of token t, nothing more:
    - the supervised model's top-8 scores for layer L+1, softmax-normalised
      (its confidence, which is the only thing that distinguishes a prediction
      worth acting on from a guess)
    - how many of its top-8 are ALREADY resident in layer L+1's cache
    - free slots available in layer L+1's pool
    - the layer index, normalised
    - the running expert hit rate

ACTION a -- how many of the ranked predictions to actually issue: {0, 1, 2, 3}.
    The ranking comes from the supervised model; the policy chooses depth. This
    is the decision supervised learning structurally cannot make, because a model
    trained on Recall@8 always wants all eight, and the measured system says the
    right answer is usually none.

REWARD r -- in milliseconds of token time, from the cost model fitted to measured
    end-to-end throughput (artifacts/cost_model.json):
        + B per correct prefetch   (converts a miss; the upload was going to
                                    happen on demand anyway, so it is not charged)
        - C per incorrect prefetch (a whole upload that would never have occurred)
    B and C are measured, not chosen, and together they set a break-even
    precision of C/(B+C) = 75%. That is the number the policy is really trading
    against: below it, issuing nothing beats issuing anything, and an agent
    rewarded on hit rate instead would confidently make the system slower.

TRANSITION -- the real engine's mechanics (PrefetchEnv): slots freed at the
    previous step boundary, uploads issued during the graph, publication at the
    step boundary, LRU refill. Prefetching evicts, so actions have consequences
    for later layers and this is a genuine sequential problem rather than a
    bandit, even though the horizon is short.

## What is actually reported

REINFORCE with a learned value baseline, against fixed-depth policies (always 0,
1, 2, 3) as controls. And then the same comparison swept over C, because the
single most useful output of this stage is not "the agent got 0.3% more" -- it is
the break-even upload cost at which prefetching starts to pay at all, which tells
you exactly which engineering change would matter.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from prefetch_env import PrefetchEnv, fit_cost_model, N_EXPERT, K  # noqa: E402
import models_deep as M  # noqa: E402
from train_deep import Index  # noqa: E402

ARTIFACTS = os.path.join(ROOT, "artifacts")
ACTIONS = [0, 1, 2, 3]
STATE_DIM = 8 + 4


class Policy(nn.Module):
    """Deliberately tiny. It runs once per layer beside the predictor, so it is
    part of the same 10 us budget, and a policy that costs more than the traffic
    it saves is self-defeating."""

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.Tanh(),
                                 nn.Linear(hidden, len(ACTIONS)))
        self.value = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 1))

    def forward(self, s):
        return self.net(s), self.value(s).squeeze(-1)

    def macs(self):
        return 2 * (STATE_DIM * 32 + 32 * len(ACTIONS))


def episode_tokens(ix, split_id, max_tokens):
    """Group index rows into tokens, in sequence order, for one split."""
    d = ix.d
    rows = ix.rows(split_id)
    order = np.lexsort((d["layer"][rows], d["pos"][rows], d["prompt"][rows]))
    rows = rows[order]
    toks, cur, key = [], [], None
    for r in rows:
        k = (int(d["prompt"][r]), int(d["pos"][r]))
        if k != key:
            if cur:
                toks.append(cur)
            cur, key = [], k
        cur.append(int(r))
    if cur:
        toks.append(cur)
    return toks[:max_tokens]


@torch.no_grad()
def rank_all(model, ix, rows, device, topn=8):
    """Precompute the supervised model's ranked predictions and softmax scores."""
    ids = np.zeros((len(rows), topn), dtype=np.int16)
    sc = np.zeros((len(rows), topn), dtype=np.float32)
    pos = {int(r): i for i, r in enumerate(rows)}
    for s in range(0, len(rows), 32768):
        idx = np.asarray(rows[s:s + 32768])
        b, _, _ = ix.batch(idx, device)
        logits, _ = model(b)
        p = torch.softmax(logits, dim=1)
        top = p.topk(topn, dim=1)
        ids[s:s + len(idx)] = top.indices.cpu().numpy().astype(np.int16)
        sc[s:s + len(idx)] = top.values.cpu().numpy()
    return ids, sc, pos


def build_next_use(ix, rows):
    """(prompt, position, layer) -> experts that layer routes to on the NEXT token.

    A prefetch issued at layer L is published at the step boundary, so the expert
    is resident from the next token onward -- never for the token that triggered
    it. Crediting it against this token's routing, which an earlier version did,
    rewards a prediction the cache could not have used and put the reward column
    in direct contradiction with the throughput column.
    """
    d = ix.d
    cur_at = {}
    for r in rows:
        cur_at[(int(d["prompt"][r]), int(d["pos"][r]), int(d["layer"][r]))] = d["cur"][r]
    return cur_at


def run_episodes(tokens, ix, ranked, policy, env_kw, cost, device, train=False,
                 fixed=None, opt=None, gamma=0.99, next_use=None):
    """Replay tokens through the engine, choosing prefetch depth at each layer."""
    ids_arr, sc_arr, pos = ranked
    env = PrefetchEnv(cost=cost, **env_kw)
    d = ix.d
    logps, values, rewards = [], [], []
    total_r = 0.0
    n_tok = 0
    for tok in tokens:
        for r in tok:
            L = int(d["layer"][r])
            nxt = L + 1
            i = pos[r]
            cand = [int(e) for e in ids_arr[i]]
            scores = sc_arr[i]
            resident_now = sum(1 for e in cand if e in env.resident[nxt])
            s = np.concatenate([
                scores,
                [resident_now / 8.0, env.free[nxt] / 8.0, L / env.n_layers,
                 env.stats["hit"] / max(env.stats["lookup"], 1)],
            ]).astype(np.float32)
            st = torch.from_numpy(s).to(device)
            if fixed is not None:
                a = fixed
                lp = v = None
            else:
                logits, v = policy(st)
                dist = torch.distributions.Categorical(logits=logits)
                a_t = dist.sample() if train else logits.argmax()
                lp = dist.log_prob(a_t)
                a = ACTIONS[int(a_t)]
            issued = env.prefetch(nxt, [e for e in cand if e not in env.resident[nxt]], a,
                                  min_keep=env.max_inserts)
            # A correct prefetch converts a miss and adds no upload (the demand
            # path would have fetched that expert anyway). An incorrect one is a
            # wasted upload. See prefetch_env's docstring for the accounting.
            # what the prefetch can actually be used for: layer L+1's routing on
            # the NEXT token, since that is when the upload is published
            nxt_ids = None
            if next_use is not None:
                nxt_ids = next_use.get((int(d["prompt"][r]), int(d["pos"][r]) + 1, nxt))
            truth = (set(int(e) for e in nxt_ids if e >= 0) if nxt_ids is not None
                     else set(int(e) for e in d["target"][r] if e >= 0))
            n_correct = sum(1 for e in issued if e in truth)
            rew = env.reward(n_correct, len(issued) - n_correct)
            rewards.append(rew)
            total_r += rew
            if lp is not None:
                logps.append(lp)
                values.append(v)
            # the layer itself runs
            cur = [int(e) for e in d["cur"][r] if e >= 0]
            env.observe(L, cur)
            env.admit_demand(L, cur)
        env.step_boundary()
        n_tok += 1

    if train and logps:
        R, returns = 0.0, []
        for rr in reversed(rewards):
            R = rr + gamma * R
            returns.append(R)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        V = torch.stack(values)
        adv = returns - V.detach()
        loss = -(torch.stack(logps) * adv).mean() + 0.5 * ((V - returns) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    tok_s, hit, up = env.tok_s(n_tok)
    return {"reward": total_r / max(n_tok, 1), "tok_s": tok_s, "hit": hit,
            "uploads_per_token": up, "n_tokens": n_tok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(ROOT, "data", "index-v3.npz"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ix = Index(args.index)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = M.build(ck["cfg"]["arch"], ix.n_layers, r=ck["cfg"].get("r", 64),
                    hidden=ck["cfg"].get("hidden", 128),
                    context=ck["cfg"].get("context", True)).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    cost = fit_cost_model()
    env_kw = {"capacity": 66, "max_inserts": 2, "pool_extra": 3, "n_layers": ix.n_layers}
    # NOTE: this environment is for RANKING policies. Its absolute throughput is
    # optimistic -- it issues 15.5 uploads a token where the engine measures 23.6
    # -- so authoritative tok/s comes from experiments/e4_config_sweep.sh.


    train_toks = episode_tokens(ix, 0, args.tokens)
    val_toks = episode_tokens(ix, 1, args.tokens)
    rows_all = np.unique(np.concatenate([np.asarray(t) for t in train_toks + val_toks]))
    ranked = rank_all(model, ix, rows_all, device)
    next_use = build_next_use(ix, np.concatenate([ix.rows(0), ix.rows(1), ix.rows(2)]))
    print(f"{len(train_toks)} train tokens, {len(val_toks)} val tokens", file=sys.stderr)

    # --- controls: fixed prefetch depth
    print("\n=== fixed-depth controls (validation) ===\n", file=sys.stderr)
    base = {}
    for a in ACTIONS:
        r = run_episodes(val_toks, ix, ranked, None, env_kw, cost, device, fixed=a,
                         next_use=next_use)
        base[a] = r
        print(f"  depth {a}: {r['tok_s']:6.1f} tok/s  hit {100 * r['hit']:.1f}%  "
              f"{r['uploads_per_token']:.1f} up/tok  reward {r['reward']:+.3f} ms/tok",
              file=sys.stderr)

    # --- REINFORCE
    policy = Policy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    best, best_state = -1e9, None
    hist = []
    for ep in range(args.episodes):
        run_episodes(train_toks, ix, ranked, policy, env_kw, cost, device,
                     train=True, opt=opt, next_use=next_use)
        v = run_episodes(val_toks, ix, ranked, policy, env_kw, cost, device,
                         next_use=next_use)
        hist.append(v["tok_s"])
        if v["reward"] > best:
            best = v["reward"]
            best_state = {k: t.detach().clone() for k, t in policy.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"  episode {ep + 1:3d}  val {v['tok_s']:6.1f} tok/s  "
                  f"reward {v['reward']:+.3f}  {v['uploads_per_token']:.2f} up/tok",
                  file=sys.stderr, flush=True)
    policy.load_state_dict(best_state)
    final = run_episodes(val_toks, ix, ranked, policy, env_kw, cost, device,
                         next_use=next_use)

    # --- the informative part: sweep the upload cost
    print("\n=== break-even: how cheap must an upload be for prefetch to pay? ===\n",
          file=sys.stderr)
    sweep = []
    C0 = cost["C_ms_per_upload"]
    for scale in (1.0, 0.5, 0.25, 0.1, 0.05, 0.0):
        c2 = dict(cost)
        c2["C_ms_per_upload"] = C0 * scale
        rows = {a: run_episodes(val_toks, ix, ranked, None, env_kw, c2, device, fixed=a,
                                next_use=next_use) for a in ACTIONS}
        rl = run_episodes(val_toks, ix, ranked, policy, env_kw, c2, device, next_use=next_use)
        bestfixed = max(rows, key=lambda a: rows[a]["tok_s"])
        sweep.append({"upload_cost_us": 1000 * C0 * scale, "scale": scale,
                      "best_fixed_depth": bestfixed,
                      "tok_s_depth0": rows[0]["tok_s"],
                      "tok_s_best_fixed": rows[bestfixed]["tok_s"],
                      "tok_s_rl": rl["tok_s"], "rl_uploads": rl["uploads_per_token"]})
        print(f"  upload {1000 * C0 * scale:6.1f} us -> best fixed depth {bestfixed}, "
              f"{rows[bestfixed]['tok_s']:6.1f} tok/s vs {rows[0]['tok_s']:6.1f} at depth 0",
              file=sys.stderr)

    out = {"cost_model": cost, "fixed": {str(a): base[a] for a in base},
           "rl": final, "history": hist, "sweep": sweep,
           "policy_macs": policy.macs(), "ckpt": os.path.basename(args.ckpt)}
    if args.test:
        test_toks = episode_tokens(ix, 2, args.tokens)
        rows_t = np.unique(np.concatenate([np.asarray(t) for t in test_toks]))
        ranked_t = rank_all(model, ix, rows_t, device)
        out["test"] = {
            "rl": run_episodes(test_toks, ix, ranked_t, policy, env_kw, cost, device,
                               next_use=next_use),
            "fixed": {str(a): run_episodes(test_toks, ix, ranked_t, None, env_kw, cost,
                                           device, fixed=a, next_use=next_use)
                      for a in ACTIONS}}
    os.makedirs(ARTIFACTS, exist_ok=True)
    json.dump(out, open(os.path.join(ARTIFACTS, "stage_d_rl.json"), "w"), indent=1, default=float)
    torch.save({"state": policy.state_dict(), "tag": "FRESH_RL"},
               os.path.join(ARTIFACTS, "ckpt", "FRESH_RL-policy.pt"))
    print(f"\nRL policy: {final['tok_s']:.1f} tok/s, {final['uploads_per_token']:.2f} up/tok; "
          f"best fixed depth {max(base, key=lambda a: base[a]['tok_s'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
