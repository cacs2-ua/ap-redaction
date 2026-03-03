from __future__ import annotations

import os
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


# ----------------------------
# Language specification: a^n b^n
# ----------------------------
@dataclass(frozen=True)
class LanguageSpec:
    token_index: Dict[str, int]
    index_token: Dict[int, str]
    vocab_size: int
    fill_soft_target: Callable[[str, int, torch.Tensor, Dict[str, int]], None]
    generate_sequence: Callable[[int], str]


def fill_soft_target_anbn(seq: str, j: int, soft_target: torch.Tensor, token_index: Dict[str, int]) -> None:
    # soft_target is a 1D tensor over vocab
    soft_target.zero_()
    if seq[j] == '^':
        soft_target[token_index['a']] = 0.5
        soft_target[token_index['$']] = 0.5
    elif seq[j] == 'a':
        soft_target[token_index['a']] = 0.5
        soft_target[token_index['b']] = 0.5
    elif seq[j] == 'b':
        soft_target[token_index[seq[j + 1]]] = 1.0
    elif seq[j] == '$':
        soft_target[token_index['$']] = 1.0


def generate_sequence_anbn(n: int) -> str:
    return '^' + 'a' * n + 'b' * n + '$'


def get_language_anbn() -> LanguageSpec:
    token_index = {'PAD': 0, '^': 1, 'a': 2, 'b': 3, 'c': 4, '$': 5}
    index_token = {v: k for k, v in token_index.items()}
    return LanguageSpec(
        token_index=token_index,
        index_token=index_token,
        vocab_size=len(token_index),
        fill_soft_target=fill_soft_target_anbn,
        generate_sequence=generate_sequence_anbn,
    )


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
    if not train:
        seqs = [language.generate_sequence(i) for i in range(batch_size)]
    else:
        seqs = [language.generate_sequence(random.randint(0, n_max_train + 1)) for _ in range(batch_size)]

    max_len = max(len(s) for s in seqs)
    inp = torch.full((batch_size, max_len), fill_value=tok['PAD'], dtype=torch.long, device=device)
    tgt = torch.full((batch_size, max_len), fill_value=tok['PAD'], dtype=torch.long, device=device)
    soft = torch.zeros((batch_size, max_len, language.vocab_size), device=device, dtype=torch.float32)

    for i, seq in enumerate(seqs):
        for j, sym in enumerate(seq):
            inp[i, j] = tok[sym]

        # targets: shifted next symbol distribution
        for j, sym_next in enumerate(seq[1:]):
            tgt[i, j] = tok[sym_next]
            language.fill_soft_target(seq, j, soft[i, j], tok)

        last = len(seq) - 1
        tgt[i, last] = tok['$']
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

    seq = "".join([itok[int(c.item())] for c in inp[row] if int(c.item()) != tok['PAD']])
    out = [f"{seq} · "]
    for i, c in enumerate(inp[row]):
        if int(c.item()) == tok['PAD']:
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
    # returns accuracy per sequence (B,)
    tok = language.token_index
    indices_winner = torch.argmax(logits, dim=-1)
    padding_mask = inp != tok['PAD']
    mask_winner = torch.gather(soft_target, dim=-1, index=indices_winner.unsqueeze(-1)).squeeze(-1).bool()
    correct = padding_mask & mask_winner
    acc = correct.int().sum(dim=-1) * 100.0 / padding_mask.int().sum(dim=-1)

    if debug_row is not None:
        print_sequence_and_predictions(language, inp, soft_target, indices_winner, debug_row, print_fn=print_fn)

    return acc


def generalization_range_from_accuracy(acc: torch.Tensor) -> int:
    # acc is (n_test+1,) where index n corresponds to the sequence with that n
    # returns largest n such that acc[n] == 100; if fails at n=0, returns -1
    success_n = int(acc.shape[0]) - 1
    for n in range(acc.shape[0]):
        if float(acc[n].item()) < 100.0:
            success_n = n - 1
            break
    return int(success_n)


# ----------------------------
# Model with optional mechanics collection
# ----------------------------
class LSTMLayer(nn.Module):
    def __init__(self, input_size: int, num_cells: int, drop_prob: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.num_cells = num_cells

        self.W_f = nn.Linear(input_size, num_cells, bias=False)
        self.U_f = nn.Linear(num_cells, num_cells, bias=False)
        self.W_g = nn.Linear(input_size, num_cells, bias=False)
        self.U_g = nn.Linear(num_cells, num_cells, bias=False)
        self.W_i = nn.Linear(input_size, num_cells, bias=False)
        self.U_i = nn.Linear(num_cells, num_cells, bias=False)
        self.W_o = nn.Linear(input_size, num_cells, bias=False)
        self.U_o = nn.Linear(num_cells, num_cells, bias=False)

        self.dropout = nn.Dropout(drop_prob)
        self.h_t: Optional[torch.Tensor] = None
        self.c_t: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, reset_states: bool = False, return_gates: bool = True):
        # x: (B, C)
        B, _ = x.shape
        if reset_states or self.h_t is None or self.c_t is None or self.h_t.shape[0] != B:
            self.h_t = torch.zeros((B, self.num_cells), device=x.device)
            self.c_t = torch.zeros((B, self.num_cells), device=x.device)

        h_t, c_t = self.h_t, self.c_t

        f = torch.sigmoid(self.W_f(x) + self.U_f(h_t))
        k = c_t * f

        g = torch.tanh(self.W_g(x) + self.U_g(h_t))
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_t))
        j = g * i

        c_t = k + j
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_t))
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
    ):
        super().__init__()
        assert num_layers >= 1, f"Invalid number of layers: {num_layers}"
        self.num_cells = num_cells
        self.num_layers = num_layers
        self.output_size = output_size

        self.embeddings = nn.Embedding(vocab_size, embd_size)
        self.lstms = nn.ModuleList(
            [LSTMLayer(embd_size, num_cells, drop_prob=drop_prob)]
            + [LSTMLayer(num_cells, num_cells, drop_prob=drop_prob) for _ in range(num_layers - 1)]
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
# Weight initialization (grid-search target)
# ----------------------------
def init_model_weights(
    model: RNN,
    scheme: str,
    seed: int,
    init_scale: float = 1.0,
    verbose: bool = False,
    print_fn=print,
) -> None:
    """
    Schemes supported:
      - "pytorch_default": do nothing
      - "xavier_uniform"
      - "xavier_normal"
      - "kaiming_uniform"
      - "kaiming_normal"
      - "orthogonal_recurrent_xavier": Xav for input weights, orthogonal for recurrent U_* (square)
      - "small_normal": Normal(0, 0.1*init_scale)
      - "small_uniform": Uniform(-0.1*init_scale, +0.1*init_scale)
    """
    set_seed(seed, deterministic=False)  # init determinism is enough; training seed is handled elsewhere

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

    # embeddings + fc
    if hasattr(model, "embeddings"):
        if scheme in ("small_normal",):
            nn.init.normal_(model.embeddings.weight, mean=0.0, std=0.1 * init_scale)
        elif scheme in ("small_uniform",):
            nn.init.uniform_(model.embeddings.weight, a=-0.1 * init_scale, b=0.1 * init_scale)
        else:
            nn.init.normal_(model.embeddings.weight, mean=0.0, std=0.02 * init_scale)

    _init_linear(model.fc, is_recurrent_square=False)

    # LSTM weights
    for layer in model.lstms:
        for name in ["W_f", "W_g", "W_i", "W_o"]:
            _init_linear(getattr(layer, name), is_recurrent_square=False)
        for name in ["U_f", "U_g", "U_i", "U_o"]:
            lin = getattr(layer, name)
            is_sq = (lin.weight.shape[0] == lin.weight.shape[1])
            _init_linear(lin, is_recurrent_square=is_sq)

    if verbose:
        print_fn(f"[init] scheme={scheme}, init_scale={init_scale}")
        print_fn(describe_tensor(model.fc.weight, "fc.weight"))
        print_fn(describe_tensor(model.embeddings.weight, "embeddings.weight"))


# ----------------------------
# Training + evaluation + mechanistic fingerprint
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
    verbosity: int = 1,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      - converged, steps_to_converge, final_train_mean_acc
      - success_n (generalization range upper bound)
      - fingerprint_* (mechanistic signature summary)
      - plus debug stats
    """
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
    ).to(device)

    # init weights (grid-search axis)
    init_model_weights(model, scheme=init_scheme, seed=seed, init_scale=init_scale, verbose=(verbosity >= 2))

    criterion = nn.CrossEntropyLoss(ignore_index=language.token_index["PAD"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=scheduler_end_factor, total_iters=training_steps
    )

    # fixed evaluation batch: sequences 0..n_max_train
    input_eval, target_eval, soft_eval = make_batch(language, n_max_train + 1, device=device, train=False)

    converged = False
    steps_to_converge = None
    last_valid_mean_acc = float("nan")
    last_loss = float("nan")

    if verbosity >= 1:
        print(f"\n[run] seed={seed} lr={learning_rate} cells={num_cells} layers={num_layers} "
              f"init={init_scheme} scale={init_scale} drop={drop_prob} embd={embd_size} device={device}")
        print("[run] " + describe_tensor(input_eval, "input_eval") + " | " + describe_tensor(soft_eval, "soft_eval"))

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

        # quick NaN check
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
                    print(f"[valid] step={step:4d} loss={last_loss:.4f} mean_acc={last_valid_mean_acc:.2f} "
                          f"lr={lr_now:.5f} grad_l2={gstats['grad_l2']:.4g} grad_max={gstats['grad_max_abs']:.4g}")

                if verbosity >= 2:
                    # occasionally print a debug sequence
                    debug_row = max(0, min(n_max_train - 1, acc.shape[0] - 1))
                    _ = evaluate(language, input_eval, soft_eval, logits_eval, debug_row=debug_row)

                if full_acc and not converged:
                    converged = True
                    steps_to_converge = step
                    if verbosity >= 1:
                        print("[valid] Training converged (100% on training range).")
                    model.train()
                    break

            model.train()

    # Generalization evaluation: test up to n_test
    model.eval()
    n_test = int(n_max_train * n_test_factor)
    input_test, target_test, soft_test = make_batch(language, n_test + 1, device=device, train=False)

    with torch.no_grad():
        logits_test, outputs, hidden_states, gates = model(input_test, collect_mechanics=True)
        acc_test = evaluate(language, input_test, soft_test, logits_test, debug_row=min(n_max_train - 1, n_test))
        success_n = generalization_range_from_accuracy(acc_test)

    # Mechanistic fingerprint on a probe sequence (default: n_probe = min(8, n_test))
    n_probe = int(min(8, n_test))
    fp = mechanistic_fingerprint_anbn(language, input_test, hidden_states, gates, n_probe=n_probe)

    elapsed = time.time() - t0

    record: Dict[str, Any] = {
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
        "converged": converged,
        "steps_to_converge": steps_to_converge if steps_to_converge is not None else -1,
        "final_train_mean_acc": last_valid_mean_acc,
        "final_loss": last_loss,
        "success_n": success_n,
        "elapsed_sec": elapsed,
    }
    record.update(fp)

    if verbosity >= 1:
        print(f"[test] generalization_range=[0,{success_n}] (tested up to {n_test}) | "
              f"fingerprint={record.get('fp_code','NA')} | elapsed={elapsed:.2f}s")

    # optional persistence for debugging
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        run_id = f"cells{num_cells}_layers{num_layers}_lr{learning_rate}_init{init_scheme}_seed{seed}"
        with open(os.path.join(out_dir, f"{run_id}.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    return record


def mechanistic_fingerprint_anbn(
    language: LanguageSpec,
    input_test: torch.Tensor,
    hidden_states: torch.Tensor,
    gates: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    n_probe: int = 8,
) -> Dict[str, Any]:
    """
    Produces a lightweight numeric+symbolic fingerprint of the mechanism for a^n b^n.

    We measure, for layer=0:
      - correlation of each cell state c_t with an expected counter value over time
      - mean gates (forget/input/output) over 'a' segment and 'b' segment
    """
    tok = language.token_index
    itok = language.index_token

    # row n_probe corresponds to sequence '^' + a*n + b*n + '$'
    row = int(max(0, min(n_probe, input_test.shape[0] - 1)))

    # Extract actual token sequence without PAD
    seq_tokens = [int(x.item()) for x in input_test[row] if int(x.item()) != tok["PAD"]]
    seq_syms = [itok[t] for t in seq_tokens]
    T = len(seq_syms)

    # layer 0 only (you can extend to multiple layers similarly)
    c = hidden_states[row, :T, 0, :]  # (T, C)
    fg, ig, og = gates
    fg = fg[row, :T, 0, :]  # (T, C)
    ig = ig[row, :T, 0, :]
    og = og[row, :T, 0, :]

    # expected counter after reading each symbol:
    # counter = (#a seen so far) - (#b seen so far)
    counter = []
    a_seen = 0
    b_seen = 0
    for s in seq_syms:
        if s == "a":
            a_seen += 1
        elif s == "b":
            b_seen += 1
        counter.append(a_seen - b_seen)
    counter_t = torch.tensor(counter, device=c.device, dtype=torch.float32)  # (T,)

    # correlation per cell
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

    # segment masks
    a_mask = torch.tensor([1.0 if s == "a" else 0.0 for s in seq_syms], device=c.device)
    b_mask = torch.tensor([1.0 if s == "b" else 0.0 for s in seq_syms], device=c.device)

    def masked_mean(mat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mat: (T,C), mask: (T,)
        denom = mask.sum().clamp(min=1.0)
        return (mat * mask.unsqueeze(-1)).sum(dim=0) / denom

    fg_a = masked_mean(fg, a_mask)[counter_cell].item() if counter_cell >= 0 else float("nan")
    ig_a = masked_mean(ig, a_mask)[counter_cell].item() if counter_cell >= 0 else float("nan")
    og_a = masked_mean(og, a_mask)[counter_cell].item() if counter_cell >= 0 else float("nan")

    fg_b = masked_mean(fg, b_mask)[counter_cell].item() if counter_cell >= 0 else float("nan")
    ig_b = masked_mean(ig, b_mask)[counter_cell].item() if counter_cell >= 0 else float("nan")
    og_b = masked_mean(og, b_mask)[counter_cell].item() if counter_cell >= 0 else float("nan")

    # symbolic code (coarse): which cell + correlation sign + rough gate regime
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


# ----------------------------
# Grid search (multi-seed) + aggregation
# ----------------------------
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
    """
    grid keys supported (typical):
      - learning_rate, num_cells, num_layers
      - init_scheme, init_scale
      - embd_size, drop_prob, batch_size, training_steps, valid_steps, n_max_train, n_test_factor

    fixed: any train_one_run kwargs you want constant.

    Returns: pandas DataFrame if pandas exists, else list[dict].
    Also writes:
      - out_dir/runs.jsonl  (one record per run)
      - out_dir/runs.csv    (if pandas available)
      - out_dir/summary.json (aggregate per config)
    """
    os.makedirs(out_dir, exist_ok=True)

    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    configs = list(itertools.product(*values))

    all_records: List[Dict[str, Any]] = []
    jsonl_path = os.path.join(out_dir, "runs.jsonl")

    if verbosity >= 1:
        print(f"[grid] configs={len(configs)} seeds_per_config={len(seeds)} total_runs={len(configs)*len(seeds)}")
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
                    out_dir=None,  # we already do a jsonl for all; set to out_dir if you want per-run JSONs too
                    **kwargs,
                )
                rec["config_id"] = ci
                for k in keys:
                    rec[f"cfg_{k}"] = cfg[k]

                all_records.append(rec)
                fjsonl.write(json.dumps(rec) + "\n")
                fjsonl.flush()

    # Aggregate by config (mean/std over seeds)
    summary = aggregate_by_config(all_records, group_keys=[f"cfg_{k}" for k in keys])

    # Save summary
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Return DataFrame if possible
    if pd is not None:
        df = pd.DataFrame(all_records)
        df.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
        return df
    else:
        # minimal CSV without pandas
        write_csv_fallback(all_records, os.path.join(out_dir, "runs.csv"))
        return all_records


def aggregate_by_config(records: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    # group by tuple(group_keys)
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
        # key metrics
        succ = [float(x["success_n"]) for x in rs]
        conv = [1.0 if bool(x["converged"]) else 0.0 for x in rs]
        stc = [float(x["steps_to_converge"]) for x in rs]
        corr = [float(x.get("fp_corr", float("nan"))) for x in rs]

        item["n_runs"] = len(rs)
        item["success_n_mean"], item["success_n_std"] = mean_std(succ)
        item["converged_rate_mean"], item["converged_rate_std"] = mean_std(conv)
        item["steps_to_converge_mean"], item["steps_to_converge_std"] = mean_std(stc)
        item["fp_corr_mean"], item["fp_corr_std"] = mean_std(corr)

        # crude measure of "same mechanism": fraction of most common fp_code
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

    # sort by success_n_mean descending then converged_rate
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
