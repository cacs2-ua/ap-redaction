from __future__ import annotations

import os
import re
import json
import math
import time
import itertools
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import pandas as pd  # optional
except Exception:
    pd = None


# ----------------------------
# Reproducibility utilities
# ----------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # NOTE: deterministic algorithms can raise errors depending on ops/GPU.
        torch.use_deterministic_algorithms(True)


def describe_tensor(x: torch.Tensor, name: str = "tensor") -> str:
    """
    Safe tensor descriptor for debugging.
    Works for integer tensors by casting to float ONLY for stats.
    """
    if x is None:
        return f"{name}: None"

    with torch.no_grad():
        x_det = x.detach()
        n = x_det.numel()
        if n == 0:
            return f"{name}: empty"

        # Cast only for stats when needed (e.g., Long tensors).
        if (not x_det.is_floating_point()) and (not x_det.is_complex()):
            x_stats = x_det.float()
        else:
            x_stats = x_det

        finite = torch.isfinite(x_stats)
        n_finite = int(finite.sum().item())

        # NaN/Inf are meaningful only in float/complex, but computing them on x_stats is safe.
        n_nan = int(torch.isnan(x_stats).sum().item()) if (x_stats.is_floating_point() or x_stats.is_complex()) else 0
        n_inf = int(torch.isinf(x_stats).sum().item()) if (x_stats.is_floating_point() or x_stats.is_complex()) else 0

        if n_finite == 0:
            return (
                f"{name}: shape={tuple(x_det.shape)}, dtype={x_det.dtype}, device={x_det.device}, "
                f"numel={n}, finite=0, nan={n_nan}, inf={n_inf}"
            )

        xf = x_stats[finite]
        x_min = float(xf.min().item())
        x_max = float(xf.max().item())
        x_mean = float(xf.mean().item())

        # Avoid NaN std for singletons
        if xf.numel() <= 1:
            x_std = 0.0
        else:
            # unbiased=False avoids edge-case NaNs and is stable for reporting
            x_std = float(xf.std(unbiased=False).item())

        return (
            f"{name}: shape={tuple(x_det.shape)}, dtype={x_det.dtype}, device={x_det.device}, "
            f"min={x_min:.4g}, max={x_max:.4g}, mean={x_mean:.4g}, std={x_std:.4g}, "
            f"nan={n_nan}, inf={n_inf}"
        )


def grad_norms(model: nn.Module) -> Dict[str, float]:
    total_sq = 0.0
    max_abs = 0.0
    n_grads = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if g.numel() == 0:
            continue
        n_grads += 1
        total_sq += float(g.pow(2).sum().item())
        max_abs = max(max_abs, float(g.abs().max().item()))
    return {
        "grad_l2": math.sqrt(total_sq) if total_sq > 0 else 0.0,
        "grad_max_abs": max_abs,
        "n_grads": float(n_grads),
    }


def param_stats(x: torch.Tensor, name: str) -> str:
    with torch.no_grad():
        if x is None:
            return f"{name}: None"
        y = x.detach().float().view(-1)
        if y.numel() == 0:
            return f"{name}: empty"
        return (
            f"{name}: shape={tuple(x.shape)} "
            f"min={float(y.min().item()):.4g} max={float(y.max().item()):.4g} "
            f"mean={float(y.mean().item()):.4g} std={float(y.std(unbiased=False).item()):.4g} "
            f"l2={float(y.pow(2).sum().sqrt().item()):.4g}"
        )


# ----------------------------
# Language specification
# ----------------------------
@dataclass(frozen=True)
class LanguageSpec:
    name: str
    token_index: Dict[str, int]
    index_token: Dict[int, str]
    vocab_size: int
    fill_soft_target: Callable[[str, int, torch.Tensor, Dict[str, int]], None]
    generate_sequence: Callable[..., str]  # signature depends on language


# --- a^n b^n ---
def fill_soft_target_anbn(seq: str, j: int, soft_target: torch.Tensor, token_index: Dict[str, int]) -> None:
    soft_target.zero_()
    if seq[j] == "^":
        # either start with 'a' or end immediately: '^$'
        soft_target[token_index["a"]] = 0.5
        soft_target[token_index["$"]] = 0.5
    elif seq[j] == "a":
        # could continue a's or switch to b's
        soft_target[token_index["a"]] = 0.5
        soft_target[token_index["b"]] = 0.5
    elif seq[j] == "b":
        # deterministic once b's start (in this task definition)
        soft_target[token_index[seq[j + 1]]] = 1.0
    elif seq[j] == "$":
        soft_target[token_index["$"]] = 1.0


def generate_sequence_anbn(n: int) -> str:
    return "^" + "a" * n + "b" * n + "$"


def get_language_anbn() -> LanguageSpec:
    token_index = {"PAD": 0, "^": 1, "a": 2, "b": 3, "c": 4, "$": 5}  # keep legacy 'c'
    index_token = {v: k for k, v in token_index.items()}
    return LanguageSpec(
        name="anbn",
        token_index=token_index,
        index_token=index_token,
        vocab_size=len(token_index),
        fill_soft_target=fill_soft_target_anbn,
        generate_sequence=generate_sequence_anbn,
    )


# --- a^n b^n c^n ---
def fill_soft_target_anbncn(seq: str, j: int, soft_target: torch.Tensor, token_index: Dict[str, int]) -> None:
    soft_target.zero_()
    if seq[j] == "^":
        soft_target[token_index["a"]] = 0.5
        soft_target[token_index["$"]] = 0.5
    elif seq[j] == "a":
        soft_target[token_index["a"]] = 0.5
        soft_target[token_index["b"]] = 0.5
    elif seq[j] == "b":
        # deterministic once b's start (boundary b->c is predictable from prefix counting)
        soft_target[token_index[seq[j + 1]]] = 1.0
    elif seq[j] == "c":
        # deterministic once c's start (boundary c->$ is predictable from prefix counting)
        soft_target[token_index[seq[j + 1]]] = 1.0
    elif seq[j] == "$":
        soft_target[token_index["$"]] = 1.0


def generate_sequence_anbncn(n: int) -> str:
    return "^" + "a" * n + "b" * n + "c" * n + "$"


def get_language_anbncn() -> LanguageSpec:
    token_index = {"PAD": 0, "^": 1, "a": 2, "b": 3, "c": 4, "$": 5}
    index_token = {v: k for k, v in token_index.items()}
    return LanguageSpec(
        name="anbncn",
        token_index=token_index,
        index_token=index_token,
        vocab_size=len(token_index),
        fill_soft_target=fill_soft_target_anbncn,
        generate_sequence=generate_sequence_anbncn,
    )

def fill_soft_target_anbmBmAn(seq: str, j: int, soft_target: torch.Tensor, token_index: Dict[str, int]) -> None:
    """
    Hybrid evaluation:
      - '^', 'a', 'b' use grammar-based plausible next symbols (non-deterministic)
      - 'B' and 'A' are deterministic because once their phases begin,
        the boundaries (B->A and A->$) are predictable from prefix counting.
    """
    soft_target.zero_()
    s = seq[j]

    # IMPORTANT: make B-phase and A-phase deterministic (counting boundaries)
    if s == "B" or s == "A":
        soft_target[token_index[seq[j + 1]]] = 1.0
        return

    if s == "^":
        opts = ["a", "b", "$"]
    elif s == "a":
        opts = ["a", "b", "A"]
    elif s == "b":
        opts = ["b", "B"]
    elif s == "$":
        opts = ["$"]
    else:
        opts = ["$"]

    p = 1.0 / max(1, len(opts))
    for o in opts:
        if o in token_index:
            soft_target[token_index[o]] = p


def generate_sequence_anbmBmAn(n: int, m: int) -> str:
    return "^" + "a" * n + "b" * m + "B" * m + "A" * n + "$"


def get_language_anbmBmAn() -> LanguageSpec:
    token_index = {"PAD": 0, "^": 1, "a": 2, "b": 3, "B": 4, "A": 5, "$": 6}
    index_token = {v: k for k, v in token_index.items()}
    return LanguageSpec(
        name="anbmBmAn",
        token_index=token_index,
        index_token=index_token,
        vocab_size=len(token_index),
        fill_soft_target=fill_soft_target_anbmBmAn,
        generate_sequence=generate_sequence_anbmBmAn,
    )


def _normalize_language_name(name: str) -> str:
    """
    Normalize language names into a compact key so that inputs like:
      - "a^n b^n"
      - "aⁿ bⁿ cⁿ"
      - "a^n b^m B^m A^n"
      - "anbmBmAn"
    map cleanly to canonical keys:
      - "anbn"
      - "anbncn"
      - "anbmbman"
    """
    s = name.strip().lower()

    # Normalize unicode superscripts commonly used in docs/pdfs
    s = s.replace("aⁿ", "a^n").replace("bⁿ", "b^n").replace("cⁿ", "c^n")
    s = s.replace("bᵐ", "b^m")

    # Replace exponent notation with compact form
    s = s.replace("a^n", "an").replace("b^n", "bn").replace("c^n", "cn").replace("b^m", "bm")

    # Remove braces and punctuation, keep only alphanumerics
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def get_language(name: str) -> LanguageSpec:
    """
    Robust language name resolver.

    Supported canonical names:
      - anbn
      - anbncn
      - anbmBmAn

    Accepts common variants like:
      "a^n b^n", "a^n b^n c^n", "a^n b^m B^m A^n"
    """
    key = _normalize_language_name(name)

    if key == "anbn":
        return get_language_anbn()

    if key == "anbncn":
        return get_language_anbncn()

    # a^n b^m B^m A^n  -> "anbmbman"
    if key == "anbmbman":
        return get_language_anbmBmAn()

    # permissive fallback: detect the 2-parameter pattern "an ... bm ... bm ... an"
    if ("an" in key) and ("bm" in key) and ("bmbm" in key) and key.endswith("an"):
        return get_language_anbmBmAn()

    raise ValueError(f"Unknown language name: {name!r}. Supported: anbn, anbncn, anbmBmAn")


# ----------------------------
# Batch generation + evaluation
# ----------------------------
def make_batch(
    language: LanguageSpec,
    batch_size: int,
    device: str,
    train: bool = False,
    n_max_train: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tok = language.token_index

    # Generate strings
    if language.name == "anbmBmAn":
        if not train:
            # evaluation batch: sweep n=0..batch_size-1 with a fixed m= n (simple)
            seqs = [language.generate_sequence(n=i, m=i) for i in range(batch_size)]
        else:
            seqs = [
                language.generate_sequence(
                    n=random.randint(0, n_max_train + 1),
                    m=random.randint(0, n_max_train + 1),
                )
                for _ in range(batch_size)
            ]
    else:
        if not train:
            seqs = [language.generate_sequence(i) for i in range(batch_size)]
        else:
            seqs = [language.generate_sequence(random.randint(0, n_max_train + 1)) for _ in range(batch_size)]

    max_len = max(len(s) for s in seqs)
    inp = torch.full((batch_size, max_len), fill_value=tok["PAD"], dtype=torch.long, device=device)
    tgt = torch.full((batch_size, max_len), fill_value=tok["PAD"], dtype=torch.long, device=device)
    soft = torch.zeros((batch_size, max_len, language.vocab_size), device=device, dtype=torch.float32)

    for i, seq in enumerate(seqs):
        for j, sym in enumerate(seq):
            inp[i, j] = tok[sym]

        # targets: shifted next symbol distribution
        for j, sym_next in enumerate(seq[1:]):
            tgt[i, j] = tok[sym_next]
            language.fill_soft_target(seq, j, soft[i, j], tok)

        last = len(seq) - 1
        tgt[i, last] = tok["$"]
        language.fill_soft_target(seq, last, soft[i, last], tok)

    return inp, tgt, soft


def print_sequence_and_predictions(
    language: LanguageSpec,
    inp: torch.Tensor,
    soft_target: torch.Tensor,
    indices_winner: torch.Tensor,
    row: int,
    print_fn=print,
) -> None:
    tok = language.token_index
    itok = language.index_token

    seq = "".join([itok[int(c.item())] for c in inp[row] if int(c.item()) != tok["PAD"]])
    out = [f"{seq} · "]
    for i, c in enumerate(inp[row]):
        if int(c.item()) == tok["PAD"]:
            break
        symbol = itok[int(c.item())]
        pred = itok[int(indices_winner[row, i].item())]
        plausible = (soft_target[row, i] > 0.0).nonzero().squeeze(-1).tolist()
        plausible_str = "/".join(itok[int(x)] for x in plausible)
        out.append(f"{symbol} {pred} {plausible_str} · ")
    print_fn("".join(out))


def evaluate(
    language: LanguageSpec,
    inp: torch.Tensor,
    soft_target: torch.Tensor,
    logits: torch.Tensor,
    debug_row: Optional[int] = None,
    print_fn=print,
) -> torch.Tensor:
    tok = language.token_index
    indices_winner = torch.argmax(logits, dim=-1)
    padding_mask = inp != tok["PAD"]
    mask_winner = torch.gather(soft_target, dim=-1, index=indices_winner.unsqueeze(-1)).squeeze(-1).bool()
    correct = padding_mask & mask_winner
    acc = correct.int().sum(dim=-1) * 100.0 / padding_mask.int().sum(dim=-1)

    if debug_row is not None:
        print_sequence_and_predictions(language, inp, soft_target, indices_winner, debug_row, print_fn=print_fn)

    return acc


def generalization_range_from_accuracy(acc: torch.Tensor) -> int:
    success_n = int(acc.shape[0]) - 1
    for n in range(acc.shape[0]):
        if float(acc[n].item()) < 100.0:
            success_n = n - 1
            break
    return int(success_n)


# ----------------------------
# LSTM with optional peepholes
# ----------------------------
class LSTMLayer(nn.Module):
    """
    Peephole connections (Gers & Schmidhuber style):
      f_t = σ(W_f x_t + U_f h_{t-1} + p_f ⊙ c_{t-1})
      i_t = σ(W_i x_t + U_i h_{t-1} + p_i ⊙ c_{t-1})
      c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
      o_t = σ(W_o x_t + U_o h_{t-1} + p_o ⊙ c_t)
      h_t = o_t ⊙ tanh(c_t)
    """

    def __init__(
        self,
        input_size: int,
        num_cells: int,
        drop_prob: float = 0.1,
        use_peepholes: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_cells = num_cells
        self.use_peepholes = bool(use_peepholes)

        self.W_f = nn.Linear(input_size, num_cells, bias=False)
        self.U_f = nn.Linear(num_cells, num_cells, bias=False)

        self.W_g = nn.Linear(input_size, num_cells, bias=False)
        self.U_g = nn.Linear(num_cells, num_cells, bias=False)

        self.W_i = nn.Linear(input_size, num_cells, bias=False)
        self.U_i = nn.Linear(num_cells, num_cells, bias=False)

        self.W_o = nn.Linear(input_size, num_cells, bias=False)
        self.U_o = nn.Linear(num_cells, num_cells, bias=False)

        # Peephole vectors (diagonal peepholes)
        if self.use_peepholes:
            self.p_f = nn.Parameter(torch.zeros(num_cells))
            self.p_i = nn.Parameter(torch.zeros(num_cells))
            self.p_o = nn.Parameter(torch.zeros(num_cells))
        else:
            # register as None-like buffers for easier introspection without parameters
            self.register_parameter("p_f", None)
            self.register_parameter("p_i", None)
            self.register_parameter("p_o", None)

        self.dropout = nn.Dropout(drop_prob)
        self.h_t: Optional[torch.Tensor] = None
        self.c_t: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, reset_states: bool = False, return_gates: bool = True):
        # x: (B, input_size)
        B, _ = x.shape
        if reset_states or self.h_t is None or self.c_t is None or self.h_t.shape[0] != B:
            self.h_t = torch.zeros((B, self.num_cells), device=x.device)
            self.c_t = torch.zeros((B, self.num_cells), device=x.device)

        h_prev, c_prev = self.h_t, self.c_t

        # forget gate
        f_pre = self.W_f(x) + self.U_f(h_prev)
        if self.use_peepholes:
            f_pre = f_pre + (c_prev * self.p_f)  # broadcast (B,C)*(C)->(B,C)
        f = torch.sigmoid(f_pre)
        k = c_prev * f

        # candidate + input gate
        g = torch.tanh(self.W_g(x) + self.U_g(h_prev))

        i_pre = self.W_i(x) + self.U_i(h_prev)
        if self.use_peepholes:
            i_pre = i_pre + (c_prev * self.p_i)
        i = torch.sigmoid(i_pre)
        j = g * i

        # cell update
        c_t = k + j

        # output gate (peephole uses *new* c_t)
        o_pre = self.W_o(x) + self.U_o(h_prev)
        if self.use_peepholes:
            o_pre = o_pre + (c_t * self.p_o)
        o = torch.sigmoid(o_pre)

        h_t = o * torch.tanh(c_t)
        h_t = self.dropout(h_t)

        self.h_t, self.c_t = h_t, c_t

        if return_gates:
            return h_t, c_t, (f, i, o)
        return h_t, c_t, None


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embd_size: int,
        num_cells: int,
        num_layers: int,
        output_size: int,
        drop_prob: float = 0.1,
        use_peepholes: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1, f"Invalid number of layers: {num_layers}"
        self.num_cells = num_cells
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_peepholes = bool(use_peepholes)

        self.embeddings = nn.Embedding(vocab_size, embd_size)
        self.lstms = nn.ModuleList(
            [LSTMLayer(embd_size, num_cells, drop_prob=drop_prob, use_peepholes=use_peepholes)]
            + [
                LSTMLayer(num_cells, num_cells, drop_prob=drop_prob, use_peepholes=use_peepholes)
                for _ in range(num_layers - 1)
            ]
        )
        self.fc = nn.Linear(num_cells, output_size)

    def forward(self, x: torch.Tensor, collect_mechanics: bool = False):
        # x: (B, T)
        B, T = x.size()
        L = self.num_layers
        C = self.num_cells

        logits = torch.zeros((B, T, self.output_size), device=x.device)

        outputs = hidden_states = forget_gates = input_gates = output_gates = None
        if collect_mechanics:
            outputs = torch.zeros((B, T, L, C), device=x.device)
            hidden_states = torch.zeros((B, T, L, C), device=x.device)
            input_gates = torch.zeros((B, T, L, C), device=x.device)
            forget_gates = torch.zeros((B, T, L, C), device=x.device)
            output_gates = torch.zeros((B, T, L, C), device=x.device)

        x = self.embeddings(x)  # (B, T, embd)

        for t in range(T):
            x_t = x[:, t, :].view(B, -1)
            for i, lstm in enumerate(self.lstms):
                h_t, c_t, gates = lstm(x_t, reset_states=(t == 0), return_gates=collect_mechanics)
                if collect_mechanics:
                    outputs[:, t, i, :] = h_t
                    hidden_states[:, t, i, :] = c_t
                    f, ig, og = gates
                    forget_gates[:, t, i, :] = f
                    input_gates[:, t, i, :] = ig
                    output_gates[:, t, i, :] = og
                x_t = h_t
            logits[:, t, :] = self.fc(x_t)

        gates_out = (forget_gates, input_gates, output_gates) if collect_mechanics else None
        return logits, outputs, hidden_states, gates_out


# ----------------------------
# Weight initialization
# ----------------------------
def init_model_weights(
    model: RNN,
    scheme: str,
    seed: int,
    init_scale: float = 1.0,
    verbose: bool = False,
    print_fn=print,
) -> None:
    set_seed(seed, deterministic=False)

    if scheme == "pytorch_default":
        if verbose:
            print_fn("[init] pytorch_default: no manual init applied.")
        return

    def _init_linear(m: nn.Linear, is_recurrent_square: bool = False) -> None:
        w = m.weight
        if scheme == "xavier_uniform":
            nn.init.xavier_uniform_(w, gain=init_scale)
        elif scheme == "xavier_normal":
            nn.init.xavier_normal_(w, gain=init_scale)
        elif scheme == "kaiming_uniform":
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            w.data.mul_(init_scale)
        elif scheme == "kaiming_normal":
            nn.init.kaiming_normal_(w, a=math.sqrt(5))
            w.data.mul_(init_scale)
        elif scheme == "small_normal":
            nn.init.normal_(w, mean=0.0, std=0.1 * init_scale)
        elif scheme == "small_uniform":
            nn.init.uniform_(w, a=-0.1 * init_scale, b=0.1 * init_scale)
        elif scheme == "orthogonal_recurrent_xavier":
            if is_recurrent_square:
                nn.init.orthogonal_(w, gain=init_scale)
            else:
                nn.init.xavier_uniform_(w, gain=init_scale)
        else:
            raise ValueError(f"Unknown init scheme: {scheme}")

        if m.bias is not None:
            nn.init.zeros_(m.bias)

    def _init_vector(p: torch.Tensor) -> None:
        # peephole vectors: keep them small
        if scheme in ("small_normal", "xavier_normal", "kaiming_normal"):
            nn.init.normal_(p, mean=0.0, std=0.1 * init_scale)
        elif scheme in ("small_uniform", "xavier_uniform", "kaiming_uniform"):
            nn.init.uniform_(p, a=-0.1 * init_scale, b=0.1 * init_scale)
        elif scheme == "orthogonal_recurrent_xavier":
            nn.init.normal_(p, mean=0.0, std=0.05 * init_scale)
        else:
            nn.init.zeros_(p)

    # embeddings + fc
    if hasattr(model, "embeddings"):
        if scheme in ("small_normal",):
            nn.init.normal_(model.embeddings.weight, mean=0.0, std=0.1 * init_scale)
        elif scheme in ("small_uniform",):
            nn.init.uniform_(model.embeddings.weight, a=-0.1 * init_scale, b=0.1 * init_scale)
        else:
            nn.init.normal_(model.embeddings.weight, mean=0.0, std=0.02 * init_scale)

    _init_linear(model.fc, is_recurrent_square=False)

    # LSTM weights (+ peepholes)
    for layer in model.lstms:
        for name in ["W_f", "W_g", "W_i", "W_o"]:
            _init_linear(getattr(layer, name), is_recurrent_square=False)
        for name in ["U_f", "U_g", "U_i", "U_o"]:
            lin = getattr(layer, name)
            is_sq = (lin.weight.shape[0] == lin.weight.shape[1])
            _init_linear(lin, is_recurrent_square=is_sq)

        if getattr(layer, "use_peepholes", False):
            _init_vector(layer.p_f.data)
            _init_vector(layer.p_i.data)
            _init_vector(layer.p_o.data)

    if verbose:
        print_fn(f"[init] scheme={scheme}, init_scale={init_scale}")
        print_fn(describe_tensor(model.fc.weight, "fc.weight"))
        print_fn(describe_tensor(model.embeddings.weight, "embeddings.weight"))
        if getattr(model, "use_peepholes", False):
            pf = model.lstms[0].p_f
            print_fn(param_stats(pf, "peephole.p_f(layer0)"))


# ----------------------------
# Mechanics helpers (language-agnostic)
# ----------------------------
def gate_means_by_symbol(
    language: LanguageSpec,
    input_seq: torch.Tensor,  # (T,)
    gates_layer0: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # each (T,C)
) -> Dict[str, Dict[str, float]]:
    """
    Returns mean gate activations for layer 0 per symbol:
      out[symbol]['f_mean'], ['i_mean'], ['o_mean']
    Aggregated over time, averaged across cells.
    """
    tok = language.token_index
    itok = language.index_token
    f, i, o = gates_layer0
    out: Dict[str, Dict[str, float]] = {}
    for t in range(input_seq.shape[0]):
        sym_id = int(input_seq[t].item())
        if sym_id == tok["PAD"]:
            break
        sym = itok[sym_id]
        if sym not in out:
            out[sym] = {"f_sum": 0.0, "i_sum": 0.0, "o_sum": 0.0, "n": 0.0}
        out[sym]["f_sum"] += float(f[t].mean().item())
        out[sym]["i_sum"] += float(i[t].mean().item())
        out[sym]["o_sum"] += float(o[t].mean().item())
        out[sym]["n"] += 1.0

    for sym, d in out.items():
        n = max(1.0, d["n"])
        out[sym] = {
            "f_mean": d["f_sum"] / n,
            "i_mean": d["i_sum"] / n,
            "o_mean": d["o_sum"] / n,
            "count": d["n"],
        }
    return out


# ----------------------------
# Fingerprints
# ----------------------------
def mechanistic_fingerprint_anbn(
    language: LanguageSpec,
    input_test: torch.Tensor,
    hidden_states: torch.Tensor,
    gates: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    n_probe: int = 8,
) -> Dict[str, Any]:
    tok = language.token_index
    itok = language.index_token

    row = int(max(0, min(n_probe, input_test.shape[0] - 1)))
    seq_tokens = [int(x.item()) for x in input_test[row] if int(x.item()) != tok["PAD"]]
    seq_syms = [itok[t] for t in seq_tokens]
    T = len(seq_syms)

    c = hidden_states[row, :T, 0, :]  # (T, C)
    fg, ig, og = gates
    fg = fg[row, :T, 0, :]
    ig = ig[row, :T, 0, :]
    og = og[row, :T, 0, :]

    counter = []
    a_seen = 0
    b_seen = 0
    for s in seq_syms:
        if s == "a":
            a_seen += 1
        elif s == "b":
            b_seen += 1
        counter.append(a_seen - b_seen)
    counter_t = torch.tensor(counter, device=c.device, dtype=torch.float32)

    corr = []
    for k in range(c.shape[1]):
        x = c[:, k].float()
        x = (x - x.mean()) / (x.std() + 1e-8)
        y = (counter_t - counter_t.mean()) / (counter_t.std() + 1e-8)
        corr_k = float((x * y).mean().item())
        corr.append(corr_k)

    abs_corr = [abs(v) for v in corr]
    counter_cell = int(np.argmax(abs_corr)) if len(abs_corr) > 0 else -1
    best_corr = float(corr[counter_cell]) if counter_cell >= 0 else float("nan")
    corr_sign = int(1 if best_corr >= 0 else -1)

    a_mask = torch.tensor([1.0 if s == "a" else 0.0 for s in seq_syms], device=c.device)
    b_mask = torch.tensor([1.0 if s == "b" else 0.0 for s in seq_syms], device=c.device)

    def masked_mean(mat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().clamp(min=1.0)
        return (mat * mask.unsqueeze(-1)).sum(dim=0) / denom

    fg_a = masked_mean(fg, a_mask)[counter_cell].item() if counter_cell >= 0 else float("nan")
    ig_a = masked_mean(ig, a_mask)[counter_cell].item() if counter_cell >= 0 else float("nan")
    og_a = masked_mean(og, a_mask)[counter_cell].item() if counter_cell >= 0 else float("nan")

    fg_b = masked_mean(fg, b_mask)[counter_cell].item() if counter_cell >= 0 else float("nan")
    ig_b = masked_mean(ig, b_mask)[counter_cell].item() if counter_cell >= 0 else float("nan")
    og_b = masked_mean(og, b_mask)[counter_cell].item() if counter_cell >= 0 else float("nan")

    gate_regime = "H" if (fg_a > 0.7 and fg_b > 0.7) else ("M" if (fg_a > 0.4 and fg_b > 0.4) else "L")
    fp_code = f"cell{counter_cell}_sign{corr_sign}_fg{gate_regime}"

    return {
        "fp_row": row,
        "fp_len": T,
        "fp_counter_cell": counter_cell,
        "fp_corr": best_corr,
        "fp_corr_sign": corr_sign,
        "fp_fg_a": float(fg_a),
        "fp_ig_a": float(ig_a),
        "fp_og_a": float(og_a),
        "fp_fg_b": float(fg_b),
        "fp_ig_b": float(ig_b),
        "fp_og_b": float(og_b),
        "fp_code": fp_code,
    }


def mechanistic_fingerprint_generic(
    language: LanguageSpec,
    input_test: torch.Tensor,
    hidden_states: torch.Tensor,
    gates: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    n_probe: int = 8,
) -> Dict[str, Any]:
    tok = language.token_index
    itok = language.index_token

    row = int(max(0, min(n_probe, input_test.shape[0] - 1)))
    seq_tokens = [int(x.item()) for x in input_test[row] if int(x.item()) != tok["PAD"]]
    T = len(seq_tokens)

    c = hidden_states[row, :T, 0, :]  # (T,C)
    fg, ig, og = gates
    fg0 = fg[row, :T, 0, :]
    ig0 = ig[row, :T, 0, :]
    og0 = og[row, :T, 0, :]

    c_abs_mean = float(c.abs().mean().item()) if c.numel() else float("nan")
    fg_mean = float(fg0.mean().item()) if fg0.numel() else float("nan")
    ig_mean = float(ig0.mean().item()) if ig0.numel() else float("nan")
    og_mean = float(og0.mean().item()) if og0.numel() else float("nan")

    fp_code = f"gen_T{T}_c{c_abs_mean:.2f}_f{fg_mean:.2f}"

    # per-symbol gate means (nice for debugging)
    input_seq = torch.tensor(seq_tokens, device=hidden_states.device, dtype=torch.long)
    per_sym = gate_means_by_symbol(
        language,
        input_seq,
        gates_layer0=(fg0, ig0, og0),
    )

    # keep JSON-friendly (no nested tensors)
    return {
        "fp_row": row,
        "fp_len": T,
        "fp_c_abs_mean": c_abs_mean,
        "fp_fg_mean": fg_mean,
        "fp_ig_mean": ig_mean,
        "fp_og_mean": og_mean,
        "fp_code": fp_code,
        "fp_gate_means_by_symbol": per_sym,
    }


def mechanistic_fingerprint(
    language: LanguageSpec,
    input_test: torch.Tensor,
    hidden_states: torch.Tensor,
    gates: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    n_probe: int = 8,
) -> Dict[str, Any]:
    if language.name == "anbn":
        return mechanistic_fingerprint_anbn(language, input_test, hidden_states, gates, n_probe=n_probe)
    return mechanistic_fingerprint_generic(language, input_test, hidden_states, gates, n_probe=n_probe)


# ----------------------------
# Training + evaluation
# ----------------------------
def train_one_run(
    language: LanguageSpec,
    *,
    seed: int,
    device: str,
    embd_size: int,
    num_cells: int,
    num_layers: int,
    drop_prob: float,
    learning_rate: float,
    batch_size: int,
    training_steps: int,
    valid_steps: int,
    n_max_train: int,
    n_test_factor: int = 10,
    init_scheme: str = "pytorch_default",
    init_scale: float = 1.0,
    scheduler_end_factor: float = 0.2,
    deterministic: bool = True,
    use_peepholes: bool = False,
    verbosity: int = 1,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    set_seed(seed, deterministic=deterministic)

    if device == "cuda" and torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    model = RNN(
        vocab_size=language.vocab_size,
        embd_size=embd_size,
        num_cells=num_cells,
        num_layers=num_layers,
        output_size=language.vocab_size,
        drop_prob=drop_prob,
        use_peepholes=use_peepholes,
    ).to(device)

    init_model_weights(model, scheme=init_scheme, seed=seed, init_scale=init_scale, verbose=(verbosity >= 2))

    criterion = nn.CrossEntropyLoss(ignore_index=language.token_index["PAD"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=scheduler_end_factor, total_iters=training_steps
    )

    input_eval, target_eval, soft_eval = make_batch(language, n_max_train + 1, device=device, train=False)

    converged = False
    steps_to_converge = None
    last_valid_mean_acc = float("nan")
    last_loss = float("nan")

    if verbosity >= 1:
        print(
            f"\n[run] lang={language.name} seed={seed} lr={learning_rate} cells={num_cells} layers={num_layers} "
            f"peephole={use_peepholes} init={init_scheme} scale={init_scale} drop={drop_prob} embd={embd_size} device={device}"
        )
        print("[run] " + describe_tensor(input_eval, "input_eval") + " | " + describe_tensor(soft_eval, "soft_eval"))
        if use_peepholes:
            l0 = model.lstms[0]
            print("[run] " + param_stats(l0.p_f, "peephole.p_f(layer0)"))
            print("[run] " + param_stats(l0.p_i, "peephole.p_i(layer0)"))
            print("[run] " + param_stats(l0.p_o, "peephole.p_o(layer0)"))

    model.train()
    for step in range(training_steps):
        inp, tgt, _ = make_batch(language, batch_size, device=device, train=True, n_max_train=n_max_train)
        logits, _, _, _ = model(inp, collect_mechanics=False)

        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
        last_loss = float(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        gstats = grad_norms(model)
        optimizer.step()
        scheduler.step()

        if not math.isfinite(last_loss):
            if verbosity >= 1:
                print(f"[warn] Non-finite loss at step={step}: {last_loss}")
                print("[warn] " + describe_tensor(logits, "logits"))
            break

        if step % valid_steps == 0 or step == training_steps - 1:
            model.eval()
            with torch.no_grad():
                logits_eval, _, _, _ = model(input_eval, collect_mechanics=False)
                acc = evaluate(language, input_eval, soft_eval, logits_eval)
                last_valid_mean_acc = float(acc.mean().item())
                full_acc = bool((acc >= 100.0).all().item())

                if verbosity >= 1:
                    lr_now = optimizer.param_groups[0]["lr"]
                    print(
                        f"[valid] step={step:4d} loss={last_loss:.4f} mean_acc={last_valid_mean_acc:.2f} "
                        f"lr={lr_now:.5f} grad_l2={gstats['grad_l2']:.4g} grad_max={gstats['grad_max_abs']:.4g}"
                    )

                if verbosity >= 2:
                    debug_row = max(0, min(n_max_train - 1, acc.shape[0] - 1))
                    _ = evaluate(language, input_eval, soft_eval, logits_eval, debug_row=debug_row)

                    if use_peepholes:
                        l0 = model.lstms[0]
                        print("[dbg] " + param_stats(l0.p_f, "peephole.p_f(layer0)"))
                        print("[dbg] " + param_stats(l0.p_i, "peephole.p_i(layer0)"))
                        print("[dbg] " + param_stats(l0.p_o, "peephole.p_o(layer0)"))

                if full_acc and not converged:
                    converged = True
                    steps_to_converge = step
                    if verbosity >= 1:
                        print("[valid] Training converged (100% on training range).")
                    model.train()
                    break

            model.train()

    model.eval()
    n_test = int(n_max_train * n_test_factor)
    input_test, target_test, soft_test = make_batch(language, n_test + 1, device=device, train=False)

    with torch.no_grad():
        logits_test, outputs, hidden_states, gates = model(input_test, collect_mechanics=True)
        acc_test = evaluate(language, input_test, soft_test, logits_test, debug_row=min(n_max_train - 1, n_test))
        success_n = generalization_range_from_accuracy(acc_test)

    n_probe = int(min(8, n_test))
    fp = mechanistic_fingerprint(language, input_test, hidden_states, gates, n_probe=n_probe)

    elapsed = time.time() - t0

    record: Dict[str, Any] = {
        "lang": language.name,
        "seed": seed,
        "learning_rate": learning_rate,
        "num_cells": num_cells,
        "num_layers": num_layers,
        "embd_size": embd_size,
        "drop_prob": drop_prob,
        "training_steps": training_steps,
        "valid_steps": valid_steps,
        "n_max_train": n_max_train,
        "n_test": n_test,
        "init_scheme": init_scheme,
        "init_scale": init_scale,
        "use_peepholes": bool(use_peepholes),
        "converged": converged,
        "steps_to_converge": steps_to_converge if steps_to_converge is not None else -1,
        "final_train_mean_acc": last_valid_mean_acc,
        "final_loss": last_loss,
        "success_n": success_n,
        "elapsed_sec": elapsed,
    }
    record.update(fp)

    if verbosity >= 1:
        print(
            f"[test] lang={language.name} peephole={use_peepholes} generalization_range=[0,{success_n}] "
            f"(tested up to {n_test}) | fp={record.get('fp_code','NA')} | elapsed={elapsed:.2f}s"
        )

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        run_id = f"{language.name}_cells{num_cells}_layers{num_layers}_lr{learning_rate}_peephole{int(use_peepholes)}_seed{seed}"
        with open(os.path.join(out_dir, f"{run_id}.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    return record


# ----------------------------
# Aggregation + grid search
# ----------------------------
def aggregate_by_config(records: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in records:
        key = tuple(r.get(k) for k in group_keys)
        groups.setdefault(key, []).append(r)

    def mean_std(vals: List[float]) -> Tuple[float, float]:
        if len(vals) == 0:
            return float("nan"), float("nan")
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return m, s

    out: List[Dict[str, Any]] = []
    for key, rs in groups.items():
        item = {k: v for k, v in zip(group_keys, key)}

        succ = [float(x["success_n"]) for x in rs]
        conv = [1.0 if bool(x["converged"]) else 0.0 for x in rs]
        stc = [float(x["steps_to_converge"]) for x in rs]
        loss = [float(x["final_loss"]) for x in rs]
        acc = [float(x["final_train_mean_acc"]) for x in rs]

        item["n_runs"] = len(rs)
        item["success_n_mean"], item["success_n_std"] = mean_std(succ)
        item["converged_rate_mean"], item["converged_rate_std"] = mean_std(conv)
        item["steps_to_converge_mean"], item["steps_to_converge_std"] = mean_std(stc)
        item["final_loss_mean"], item["final_loss_std"] = mean_std(loss)
        item["final_train_acc_mean"], item["final_train_acc_std"] = mean_std(acc)

        codes = [x.get("fp_code", "NA") for x in rs]
        if len(codes) > 0:
            unique, counts = np.unique(codes, return_counts=True)
            idx = int(np.argmax(counts))
            item["fp_mode_code"] = str(unique[idx])
            item["fp_mode_frac"] = float(counts[idx] / len(codes))
        else:
            item["fp_mode_code"] = "NA"
            item["fp_mode_frac"] = float("nan")

        out.append(item)

    out.sort(key=lambda d: (d.get("success_n_mean", -1e9), d.get("converged_rate_mean", -1e9)), reverse=True)
    return out


def write_csv_fallback(records: List[Dict[str, Any]], path: str) -> None:
    import csv

    if not records:
        return
    keys = sorted({k for r in records for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow(r)


def grid_search(
    language: LanguageSpec,
    *,
    device: str,
    grid: Dict[str, List[Any]],
    seeds: List[int],
    fixed: Dict[str, Any],
    out_dir: str = "grid_runs",
    verbosity: int = 1,
) -> Any:
    os.makedirs(out_dir, exist_ok=True)

    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    configs = list(itertools.product(*values))

    all_records: List[Dict[str, Any]] = []
    jsonl_path = os.path.join(out_dir, "runs.jsonl")

    if verbosity >= 1:
        print(
            f"[grid] lang={language.name} configs={len(configs)} seeds_per_config={len(seeds)} total_runs={len(configs)*len(seeds)}"
        )
        print(f"[grid] writing run records to: {jsonl_path}")

    with open(jsonl_path, "w", encoding="utf-8") as fjsonl:
        for ci, cfg_vals in enumerate(configs):
            cfg = dict(zip(keys, cfg_vals))
            for seed in seeds:
                kwargs = dict(fixed)
                kwargs.update(cfg)

                rec = train_one_run(
                    language,
                    seed=seed,
                    device=device,
                    verbosity=verbosity,
                    out_dir=None,
                    **kwargs,
                )
                rec["config_id"] = ci
                for k in keys:
                    rec[f"cfg_{k}"] = cfg[k]

                all_records.append(rec)
                fjsonl.write(json.dumps(rec) + "\n")
                fjsonl.flush()

    summary = aggregate_by_config(all_records, group_keys=[f"cfg_{k}" for k in keys])
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if pd is not None:
        df = pd.DataFrame(all_records)
        df.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
        return df
    else:
        write_csv_fallback(all_records, os.path.join(out_dir, "runs.csv"))
        return all_records


# ----------------------------
# Peephole-vs-base comparison helpers
# ----------------------------
def compare_peepholes_vs_base(
    language: LanguageSpec,
    *,
    device: str,
    seeds: List[int],
    fixed: Dict[str, Any],
    verbosity: int = 1,
) -> Any:
    """
    Runs the same hyperparams for multiple seeds with:
      use_peepholes = False and True,
    then returns per-variant mean/std summary.
    """
    all_records: List[Dict[str, Any]] = []
    for use_p in [False, True]:
        for seed in seeds:
            rec = train_one_run(
                language,
                seed=seed,
                device=device,
                verbosity=verbosity,
                use_peepholes=use_p,
                **fixed,
            )
            all_records.append(rec)

    summary = aggregate_by_config(all_records, group_keys=["use_peepholes"])
    if verbosity >= 1:
        print("\n[compare] Summary (mean/std over seeds):")
        for s in summary:
            print(
                f"  peephole={s['use_peepholes']} | "
                f"success_n={s['success_n_mean']:.2f}±{s['success_n_std']:.2f} | "
                f"converged_rate={s['converged_rate_mean']:.2f}±{s['converged_rate_std']:.2f} | "
                f"steps={s['steps_to_converge_mean']:.1f}±{s['steps_to_converge_std']:.1f} | "
                f"loss={s['final_loss_mean']:.3f}±{s['final_loss_std']:.3f}"
            )

    if pd is not None:
        return pd.DataFrame(summary)
    return summary


# ----------------------------
# Language suite experiments
# ----------------------------
def list_language_specs() -> List[LanguageSpec]:
    """
    Canonical list of languages to explore for the assignment.
    """
    return [
        get_language_anbn(),
        get_language_anbmBmAn(),
        get_language_anbncn(),
    ]


def run_language_suite(
    *,
    device: str,
    seeds: List[int],
    fixed: Dict[str, Any],
    languages: Optional[List[LanguageSpec]] = None,
    peephole_variants: Tuple[bool, ...] = (False, True),
    verbosity: int = 1,
) -> Tuple[Any, Any]:
    """
    Runs: languages × peephole_variants × seeds

    Returns:
      - run_records: DataFrame (if pandas available) or list[dict]
      - summary:     DataFrame (if pandas available) or list[dict]
        grouped by ['lang', 'use_peepholes'] with mean/std fields.

    Debug prints:
      - Progress lines per run (when verbosity>=1)
      - Final summary + automatic difficulty ranking (when verbosity>=1)
    """
    if languages is None:
        languages = list_language_specs()

    all_records: List[Dict[str, Any]] = []
    total = len(languages) * len(peephole_variants) * len(seeds)
    done = 0

    if verbosity >= 1:
        print(f"[suite] device={device} languages={[L.name for L in languages]} seeds={seeds} peepholes={peephole_variants}")
        print(f"[suite] total_runs={total}")
        print(f"[suite] fixed_hparams={fixed}")

    for L in languages:
        if verbosity >= 1:
            print("\n" + "=" * 90)
            print(f"[suite] LANGUAGE={L.name}")

        for use_p in peephole_variants:
            if verbosity >= 1:
                print(f"[suite] variant use_peepholes={use_p}")

            for seed in seeds:
                done += 1
                if verbosity >= 1:
                    print(f"[suite] run {done}/{total} lang={L.name} peephole={use_p} seed={seed}")

                rec = train_one_run(
                    L,
                    seed=seed,
                    device=device,
                    use_peepholes=use_p,
                    verbosity=max(0, verbosity - 1),  # keep suite readable
                    **fixed,
                )
                all_records.append(rec)

    summary = aggregate_by_config(all_records, group_keys=["lang", "use_peepholes"])

    # Difficulty ranking heuristic (by success_n_mean, tie-breaker by steps_to_converge_mean)
    ranking = sorted(
        summary,
        key=lambda d: (d.get("success_n_mean", -1e9), -d.get("steps_to_converge_mean", 1e9)),
        reverse=True,
    )

    if verbosity >= 1:
        print("\n" + "-" * 90)
        print("[suite] SUMMARY (mean±std over seeds) grouped by (lang, use_peepholes):")
        for s in summary:
            print(
                f"  lang={s['lang']:<8} peephole={str(s['use_peepholes']):<5} | "
                f"success_n={s['success_n_mean']:.2f}±{s['success_n_std']:.2f} | "
                f"converged_rate={s['converged_rate_mean']:.2f}±{s['converged_rate_std']:.2f} | "
                f"steps={s['steps_to_converge_mean']:.1f}±{s['steps_to_converge_std']:.1f} | "
                f"loss={s['final_loss_mean']:.3f}±{s['final_loss_std']:.3f}"
            )

        print("\n[suite] DIFFICULTY RANKING (higher success_n_mean => easier to generalize):")
        for i, s in enumerate(ranking, 1):
            print(
                f"  #{i:02d} lang={s['lang']:<8} peephole={str(s['use_peepholes']):<5} "
                f"success_n_mean={s['success_n_mean']:.2f} converged_rate_mean={s['converged_rate_mean']:.2f}"
            )

        print("\n[suite] Interpretation rule-of-thumb for the report:")
        print("  - 'Learnable' (in this setup) ≈ high converged_rate and success_n >= n_max_train (or clearly above it).")
        print("  - 'Easier' ≈ fewer steps_to_converge AND higher success_n_mean across seeds.")
        print("  - If results vary a lot across seeds (high std), call out sensitivity to initialization/hparams.")

    if pd is not None:
        return pd.DataFrame(all_records), pd.DataFrame(summary)
    return all_records, summary

def _to_dataframe(run_records: Any):
    if pd is not None and isinstance(run_records, pd.DataFrame):
        return run_records.copy()
    if pd is not None and isinstance(run_records, list):
        return pd.DataFrame(run_records)
    # pandas not available -> keep as list[dict]
    return run_records


def _as_numpy(x) -> np.ndarray:
    return np.asarray(list(x), dtype=float)


def _probability_of_superiority(a: np.ndarray, b: np.ndarray) -> float:
    # P(a>b) + 0.5 P(a==b)
    # Uses all pairwise comparisons (small n: seeds)
    if a.size == 0 or b.size == 0:
        return float("nan")
    aa = a.reshape(-1, 1)
    bb = b.reshape(1, -1)
    win = (aa > bb).mean()
    tie = (aa == bb).mean()
    return float(win + 0.5 * tie)


def _bootstrap_ci(
    rng: np.random.Generator,
    a: np.ndarray,
    b: np.ndarray,
    stat_fn,
    n_iter: int,
    alpha: float,
) -> Tuple[float, float, float]:
    # returns: (stat_obs, ci_low, ci_high)
    stat_obs = float(stat_fn(a, b))
    if a.size == 0 or b.size == 0:
        return stat_obs, float("nan"), float("nan")

    boots = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        sa = a[rng.integers(0, a.size, size=a.size)]
        sb = b[rng.integers(0, b.size, size=b.size)]
        boots[i] = float(stat_fn(sa, sb))

    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return stat_obs, lo, hi


def _permutation_pvalue(
    rng: np.random.Generator,
    a: np.ndarray,
    b: np.ndarray,
    stat_fn,
    n_perm: int,
) -> float:
    # two-sided permutation p-value
    if a.size == 0 or b.size == 0:
        return float("nan")
    obs = float(stat_fn(a, b))
    pooled = np.concatenate([a, b], axis=0)
    n_a = a.size

    more_extreme = 0
    for _ in range(n_perm):
        idx = rng.permutation(pooled.size)
        pa = pooled[idx[:n_a]]
        pb = pooled[idx[n_a:]]
        val = float(stat_fn(pa, pb))
        if abs(val) >= abs(obs):
            more_extreme += 1

    return float((more_extreme + 1) / (n_perm + 1))


def significance_suite_by_language(
    run_records: Any,
    *,
    metric: str = "success_n",
    higher_is_better: bool = True,
    group_field: str = "use_peepholes",
    group_a_value: Any = True,
    group_b_value: Any = False,
    methods: Tuple[str, ...] = ("aso", "bootstrap", "permutation"),
    confidence_level: float = 0.95,
    num_bootstrap_iterations: int = 1000,   # used for ASO CI + mean-diff CI
    dt: float = 0.005,                      # kept for API compatibility (not required here)
    num_samples: int = 5000,                # used for permutation n_perm
    num_jobs: int = 1,                      # kept for API compatibility (single-process here)
    seed: int = 0,
    verbose: int = 0,
) -> Any:
    """
    Per-language comparison of group A vs group B.

    Implemented stats (small-sample friendly):
      - 'aso': Probability of superiority (PS) and its scaled effect: ASO = 2*PS - 1, in [-1,1]
      - 'bootstrap': bootstrap CI for mean difference (A-B)
      - 'permutation': permutation p-value for mean difference (A-B)

    Returns:
      pandas.DataFrame if pandas available, else list[dict]
    """
    alpha = 1.0 - float(confidence_level)
    rng = np.random.default_rng(int(seed))

    df = _to_dataframe(run_records)

    # If pandas not available, operate on list[dict]
    if pd is None:
        # group by lang
        by_lang: Dict[str, List[Dict[str, Any]]] = {}
        for r in df:
            by_lang.setdefault(str(r.get("lang")), []).append(r)

        out_rows: List[Dict[str, Any]] = []
        for lang, rows in by_lang.items():
            a_vals = _as_numpy([r.get(metric) for r in rows if r.get(group_field) == group_a_value])
            b_vals = _as_numpy([r.get(metric) for r in rows if r.get(group_field) == group_b_value])
            out_rows.append(_significance_one_lang(
                rng, lang, metric, a_vals, b_vals, higher_is_better, methods,
                alpha, num_bootstrap_iterations, num_samples, verbose
            ))
        return out_rows

    # pandas path
    assert pd is not None
    out_rows: List[Dict[str, Any]] = []
    for lang in sorted(df["lang"].unique().tolist()):
        sub = df[df["lang"] == lang]
        a_vals = _as_numpy(sub[sub[group_field] == group_a_value][metric].tolist())
        b_vals = _as_numpy(sub[sub[group_field] == group_b_value][metric].tolist())
        out_rows.append(_significance_one_lang(
            rng, str(lang), metric, a_vals, b_vals, higher_is_better, methods,
            alpha, num_bootstrap_iterations, num_samples, verbose
        ))

    return pd.DataFrame(out_rows)


def _significance_one_lang(
    rng: np.random.Generator,
    lang: str,
    metric: str,
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    higher_is_better: bool,
    methods: Tuple[str, ...],
    alpha: float,
    n_boot: int,
    n_perm: int,
    verbose: int,
) -> Dict[str, Any]:
    def mean_diff(a, b) -> float:
        return float(np.mean(a) - np.mean(b))

    mean_a = float(np.mean(a_vals)) if a_vals.size else float("nan")
    mean_b = float(np.mean(b_vals)) if b_vals.size else float("nan")
    diff = float(mean_a - mean_b) if (a_vals.size and b_vals.size) else float("nan")

    if not higher_is_better and np.isfinite(diff):
        # If lower is better, invert interpretation (but keep diff = mean(A)-mean(B))
        better = "A" if diff < 0 else ("B" if diff > 0 else "tie")
    else:
        better = "A" if diff > 0 else ("B" if diff < 0 else "tie")

    row: Dict[str, Any] = {
        "lang": lang,
        "metric": metric,
        "n_A": int(a_vals.size),
        "n_B": int(b_vals.size),
        "mean_A": mean_a,
        "mean_B": mean_b,
        "mean_diff_A_minus_B": diff,
        "better": better,
    }

    if verbose >= 1:
        print(f"[sig] lang={lang} metric={metric} nA={a_vals.size} nB={b_vals.size} meanA={mean_a:.3g} meanB={mean_b:.3g} diff={diff:.3g}")

    for m in methods:
        m = str(m).lower().strip()

        if m == "aso":
            ps = _probability_of_superiority(a_vals, b_vals)
            aso = float(2.0 * ps - 1.0) if np.isfinite(ps) else float("nan")

            # bootstrap CI for ASO
            def aso_stat(a, b) -> float:
                p = _probability_of_superiority(a, b)
                return float(2.0 * p - 1.0)

            aso_obs, aso_lo, aso_hi = _bootstrap_ci(rng, a_vals, b_vals, aso_stat, n_iter=n_boot, alpha=alpha)
            row.update({
                "aso_effect": float(aso_obs),
                "aso_ci_low": float(aso_lo),
                "aso_ci_high": float(aso_hi),
            })

        elif m == "bootstrap":
            diff_obs, diff_lo, diff_hi = _bootstrap_ci(rng, a_vals, b_vals, mean_diff, n_iter=n_boot, alpha=alpha)
            # “bootstrap p-value” (two-sided, sign test on bootstrapped diffs)
            # p ≈ 2*min(P(diff<=0), P(diff>=0))
            if a_vals.size and b_vals.size:
                boots = np.empty(n_boot, dtype=float)
                for i in range(n_boot):
                    sa = a_vals[rng.integers(0, a_vals.size, size=a_vals.size)]
                    sb = b_vals[rng.integers(0, b_vals.size, size=b_vals.size)]
                    boots[i] = float(np.mean(sa) - np.mean(sb))
                p_two = 2.0 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
                p_two = min(1.0, p_two)
            else:
                p_two = float("nan")

            row.update({
                "bootstrap_mean_diff": float(diff_obs),
                "bootstrap_ci_low": float(diff_lo),
                "bootstrap_ci_high": float(diff_hi),
                "bootstrap_pvalue_two_sided": float(p_two),
            })

        elif m == "permutation":
            p = _permutation_pvalue(rng, a_vals, b_vals, mean_diff, n_perm=n_perm)
            row.update({
                "permutation_pvalue_two_sided": float(p),
            })

        else:
            raise ValueError(f"Unknown method {m!r}. Supported: 'aso', 'bootstrap', 'permutation'")

    return row
