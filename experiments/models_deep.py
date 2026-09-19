"""
Model family for expert prediction, constrained by the latency budget first.

The budget is not a detail, it is the binding constraint, and it is worth
computing before choosing an architecture rather than after. At the measured
operating point the engine runs ~105 tok/s over 48 layers, so one layer is

    1 / 105 / 48  =  198 us

and the predictor runs once per layer, inside that window, on one CPU thread that
the expert FFN also wants. Spending 5% of a layer on prediction means **~10 us**,
which at a scalar ~2 GFLOP/s is roughly **20,000 multiply-accumulates**. That
number kills most of the obvious answers:

    ridge on multi-hot (the draft)   16 gathers x 128            ~2K   ok
    low-rank bilinear, r=64          16x64 + 64x128              ~9K   ok
    MLP, 256->256->128 dense         256x256 + 256x128           ~98K  5x over
    MLP, 512 hidden                                              ~200K 10x over

So "use a bigger network" is not available, and an architecture search that
ignores this would happily report a better Recall@8 for a model that makes the
system slower -- which is the specific failure the brief warns about.

What IS available is spending the budget better. The draft used 47 independent
ridge matrices, one per layer, sharing nothing: 1.5M parameters that never learn
that expert routing has common structure. Every model here instead shares a
trunk across layers and keeps only the output layer-specific, which is both
fewer parameters and more of them trained on more data.

The input is deliberately sparse. `prev` and `cur` are 8-hot, so the first layer
is 16 row-gathers, never a matrix multiply -- the same trick the draft's C++
loader already used, made architectural.

Models:
  Linear        per-layer linear on [prev|cur]. The draft, reimplemented in torch
                so the comparison is like-for-like.
  LowRank       shared sparse encoder -> per-layer decoder. ~9K MAC.
  ResMLP        shared encoder -> residual block -> per-layer decoder, with a
                learned layer embedding and optional temporal context. ~20-35K MAC.
  + confidence  every model can emit a scalar confidence, used later by the RL
                stage to decide HOW MANY experts to prefetch rather than always 8.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_EXPERT = 128
K = 8


def sparse_sum(emb, ids, mask=None):
    """Sum embedding rows for 8-hot ids. ids: (B, k) with -1 padding.

    This is the whole reason the models here are affordable: the input layer
    costs k gathers, not N_EXPERT x d multiply-adds.
    """
    safe = ids.clamp(min=0)
    out = emb(safe)                               # (B, k, d)
    valid = (ids >= 0).unsqueeze(-1).to(out.dtype)
    return (out * valid).sum(1)


class Linear(nn.Module):
    """The draft's shape: an independent map per layer, nothing shared."""

    def __init__(self, n_layers, n_expert=N_EXPERT, **kw):
        super().__init__()
        self.n_layers = n_layers
        self.w_prev = nn.Embedding(n_layers * n_expert, n_expert)
        self.w_cur = nn.Embedding(n_layers * n_expert, n_expert)
        nn.init.zeros_(self.w_prev.weight)
        nn.init.zeros_(self.w_cur.weight)
        self.bias = nn.Parameter(torch.zeros(n_layers, n_expert))

    def forward(self, batch):
        L = batch["layer"]
        off = (L * N_EXPERT).unsqueeze(1)
        s = sparse_sum(self.w_prev, torch.where(batch["prev"] >= 0, batch["prev"] + off, batch["prev"]))
        s = s + sparse_sum(self.w_cur, torch.where(batch["cur"] >= 0, batch["cur"] + off, batch["cur"]))
        return s + self.bias[L], None

    def macs(self):
        return 2 * K * N_EXPERT


class LinearCtx(nn.Module):
    """Per-layer linear, with the two extra context sets the host already has.

    The smoke run said something worth taking seriously: plain per-layer linear
    beat the shared low-rank model by six points. Expert identity is strongly
    layer-specific, and factorising across layers throws that away faster than
    sharing data recovers it.

    So this keeps the per-layer structure and spends the remaining budget on
    inputs instead of on depth: the experts layer L-1 used this token, and the
    ones layer L used on the previous token. Two more 8-hot gathers -- 4,096 MAC
    in total, a fifth of the budget -- and no matrix multiply anywhere.
    """

    def __init__(self, n_layers, n_expert=N_EXPERT, context=True, **kw):
        super().__init__()
        self.n_layers, self.context = n_layers, context
        names = ["prev", "cur"] + (["below", "self_prev"] if context else [])
        self.names = names
        self.tables = nn.ModuleDict(
            {n: nn.Embedding(n_layers * n_expert, n_expert) for n in names})
        for t in self.tables.values():
            nn.init.zeros_(t.weight)
        self.bias = nn.Parameter(torch.zeros(n_layers, n_expert))

    def forward(self, batch):
        L = batch["layer"]
        off = (L * N_EXPERT).unsqueeze(1)
        s = self.bias[L]
        for n in self.names:
            ids = batch[n]
            s = s + sparse_sum(self.tables[n], torch.where(ids >= 0, ids + off, ids))
        return s, None

    def macs(self):
        return len(self.names) * K * N_EXPERT


class LowRank(nn.Module):
    """Shared sparse encoder into r dims, then a per-layer r x 128 decoder.

    This is the draft factorised: the same per-layer expressiveness, but the
    input side is learned once from every layer's data instead of 47 times from
    a 47th of it.
    """

    def __init__(self, n_layers, r=64, n_expert=N_EXPERT, **kw):
        super().__init__()
        self.r = r
        self.e_prev = nn.Embedding(n_expert, r)
        self.e_cur = nn.Embedding(n_expert, r)
        self.layer_emb = nn.Embedding(n_layers, r)
        self.dec = nn.Parameter(torch.zeros(n_layers, r, n_expert))
        self.bias = nn.Parameter(torch.zeros(n_layers, n_expert))
        for m in (self.e_prev, self.e_cur, self.layer_emb):
            nn.init.normal_(m.weight, std=0.05)
        nn.init.normal_(self.dec, std=0.05)

    def encode(self, batch):
        return (sparse_sum(self.e_prev, batch["prev"])
                + sparse_sum(self.e_cur, batch["cur"])
                + self.layer_emb(batch["layer"]))

    def forward(self, batch):
        h = self.encode(batch)
        L = batch["layer"]
        logits = torch.bmm(h.unsqueeze(1), self.dec[L]).squeeze(1) + self.bias[L]
        return logits, None

    def macs(self):
        return 2 * K * self.r + self.r * N_EXPERT


class ResMLP(nn.Module):
    """Shared encoder, one pre-norm residual block, per-layer low-rank decoder.

    Extra temporal context is optional and costs only gathers: the experts this
    token used at layer L-1, and the ones layer L used on the previous token.
    Both are already on the host when the prediction is made.
    """

    def __init__(self, n_layers, r=64, hidden=128, n_expert=N_EXPERT,
                 context=True, confidence=True, dropout=0.0, **kw):
        super().__init__()
        self.r, self.context, self.use_conf = r, context, confidence
        self.e_prev = nn.Embedding(n_expert, r)
        self.e_cur = nn.Embedding(n_expert, r)
        self.layer_emb = nn.Embedding(n_layers, r)
        if context:
            self.e_below = nn.Embedding(n_expert, r)
            self.e_self_prev = nn.Embedding(n_expert, r)
        self.norm = nn.LayerNorm(r)
        self.fc1 = nn.Linear(r, hidden)
        self.fc2 = nn.Linear(hidden, r)
        self.drop = nn.Dropout(dropout)
        self.dec = nn.Parameter(torch.zeros(n_layers, r, n_expert))
        self.bias = nn.Parameter(torch.zeros(n_layers, n_expert))
        self.conf = nn.Linear(r, 1) if confidence else None
        for m in [self.e_prev, self.e_cur, self.layer_emb] + (
                [self.e_below, self.e_self_prev] if context else []):
            nn.init.normal_(m.weight, std=0.05)
        nn.init.normal_(self.dec, std=0.05)

    def forward(self, batch):
        h = (sparse_sum(self.e_prev, batch["prev"])
             + sparse_sum(self.e_cur, batch["cur"])
             + self.layer_emb(batch["layer"]))
        if self.context:
            h = h + sparse_sum(self.e_below, batch["below"])
            h = h + sparse_sum(self.e_self_prev, batch["self_prev"])
        h = h + self.drop(self.fc2(F.gelu(self.fc1(self.norm(h)))))
        L = batch["layer"]
        logits = torch.bmm(h.unsqueeze(1), self.dec[L]).squeeze(1) + self.bias[L]
        return logits, (self.conf(h).squeeze(-1) if self.conf is not None else None)

    def macs(self):
        n_in = 4 if self.context else 2
        return (n_in * K * self.r + self.r * self.fc1.out_features * 2
                + self.r * N_EXPERT)


ARCHS = {"linear": Linear, "linearctx": LinearCtx, "lowrank": LowRank,
         "resmlp": ResMLP}


def build(arch, n_layers, **kw):
    return ARCHS[arch](n_layers, **kw)


def loss_fn(logits, target_multihot, kind="bce", conf=None, conf_target=None):
    """`bce` treats it as 128 independent labels; `listnet` as a distribution over
    experts. Ridge used squared error on the multi-hot, which is `mse` here and is
    included so the draft's objective is one of the things being compared rather
    than an unexamined inheritance."""
    if kind == "bce":
        loss = F.binary_cross_entropy_with_logits(logits, target_multihot)
    elif kind == "mse":
        loss = F.mse_loss(logits, target_multihot)
    elif kind == "listnet":
        tgt = target_multihot / target_multihot.sum(1, keepdim=True).clamp(min=1)
        loss = -(tgt * F.log_softmax(logits, dim=1)).sum(1).mean()
    else:
        raise ValueError(kind)
    if conf is not None and conf_target is not None:
        loss = loss + 0.1 * F.mse_loss(conf, conf_target)
    return loss
