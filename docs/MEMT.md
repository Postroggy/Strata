# MemT-1.0 on strata-q36

This fork runs **MemT-1.0-Nano** (Qwen3.6-35B-A3B backbone + MemT's Native Memory
Block) on Strata's `qwen35moe` engine: the GGUF loader, the native MMVQ prompts
and decode graphs, and the long-lived session state, with the memory block ported
from `memt_sglang` (`QwenMemoryBlock`, the stable GDN-2 recurrence and the
`memt-memory-v1` snapshot).

Status: **CPU-verified only** as of 2026-11-24. The CUDA kernels are reviewed and
their math/indexing is checked on the CPU (`memt_kernels_cpu_test`), but no build
has run on a GPU and no real Nano checkpoint has been converted here (the two open
gates). Do not read this document as a speed claim.

On a GPU host, follow [MEMT-GPU-RUNBOOK.md](MEMT-GPU-RUNBOOK.md): machine
prerequisites, where the weights and artefacts go, the build/conversion/serve
commands, the three judgment points (load, P6, P7) and what to tune when the
card cannot hold every routed expert.

## What the memory block is

Every full-attention layer (`blk.3`, `blk.7`, …, `blk.39`) carries six tensors:

| GGUF tensor | shape (GGUF order) | type |
| --- | --- | --- |
| `blk.<i>.memt.in_proj.weight` | `[2048, 20992]` | Q8_0 |
| `blk.<i>.memt.f_down.weight` | `[256, 4096]` | F32 |
| `blk.<i>.memt.g_down.weight` | `[256, 4096]` | F32 |
| `blk.<i>.memt.decay_scale_log` | `[16]` | F32 |
| `blk.<i>.memt.decay_bias` | `[4096]` | F32 |
| `blk.<i>.memt.o_norm.weight` | `[256]` | F32 |

The in_proj rows are `q | k | v | erase_raw | write_raw | decay_latent |
og_latent` (4096 each, then 256 + 256). The layer reads the session's
`[16][256][256]` FP32 state (**4 MiB**, one per memory layer), adds the gated read to the attention output (the `o_proj` input) and, on a commit,
advances the state. `memt.*` metadata (`version`, `head_count`, `head_dim`,
`gate_rank`, `chunk_tokens`, `output_gate_softcap_limit`, `rms_norm_epsilon`,
`full_attention_interval`, `memory_layers`) is written by the converter and
validated by `geometry_from_gguf`.

The read snapshot is **one state per call**: every token of a request reads the
state the request began with, exactly as memt_sglang's read plan maps each chunk
to the sample's block. A query never moves the state; a commit freezes a copy at
its start, reads that copy for all tokens, and advances the live state in order
(the 2048-token write events only bound memt_sglang's internal WY chunking).

`--batch` and `--mtp` are allowed with a MemT file: the memory is per session, and
a row window (a batch step, an MTP verify window) reads each row's own session
state (§"Memory state through `terminal/serve/server.py`" below).

## Build

No geometry preset is needed: the `qwen35moe` driver reads every shape at run
time (hidden 2048, 40 layers, 16:2 heads, 32 GDN value heads, 256 experts, ff
512), and `strata` dispatches a `qwen35moe` file to it before the Flash-Next
`compiled_geometry_mismatch` runs. `-DSTRATA_GEOMETRY=q4_lite` is for the
synthetic Qwen4 family (`family = Qwen4Exp`), not for this model. The only
compile-time constraint shared kernels add is
`ssm_state_size == kCompiledGeometry.ssm_state_size` (128 in the default build,
which Nano has).

```sh
cmake -S engine -B build/cuda -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/cuda --target strata-q36 -j
```

The source directory is `engine`, not the repository root. `native` builds for the
cards in this machine; without any architecture option the build is sm_120 (RTX 50)
only and will not run on an RTX 40 card — pass `89` for a 4090.

`strata-q36:load` still refuses a shape its kernels do not take (head_dim other
than 256, state other than 128, conv other than 4, more than 512 experts, top-k
above 15).

## Convert a checkpoint

```sh
python terminal/tools/memt_to_gguf.py \
    --checkpoint /models/MemT-1.0-Nano \
    --tokenizer-from /models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --out /models/MemT-1.0-Nano.memt.gguf
```

`--checkpoint` is the full HF directory (config, index, all safetensors shards,
`tokenizer.json`, chat template); `--tokenizer-from` is a qwen35moe GGUF with the
backbone's vocabulary (any Unsloth Qwen3.6-35B-A3B-MTP quant: the converter
copies only its `tokenizer.*` fields, so a smaller quant is enough).  The
converter prints the `memt-memory-v1` identity it wrote, or a warning if the
checkpoint lacks the template/tokenizer the identity hashes.

Backbone, projections, experts and embeddings become **Q8_0**; norms, router,
`ssm_a/dt`, conv and the memory block's gate/norm tensors are **F32**, except
`memt.in_proj` (a matmul, also Q8_0). The engine's native MMVQ and grouped expert
kernels take Q8_0, not F16/BF16, and refuse a file that carries anything else. The audit tool reports the
mapping and every transform without the weights:

```sh
python terminal/tools/memt_audit.py --config .../config.json --index .../model.safetensors.index.json
```

## Serve: query, commit, snapshot

`strata-q36 --model <gguf> --serve` speaks Strata's line protocol plus:

```
MEMQUERY <max_new> [key=value ...] <id,id,...>   read the published memory, generate
MEMCOMMIT <id,id,...>                           write the prompt into memory (no generation)
MEMSELECT <session>                             the session the following MEMCOMMIT / MEMSNAP /
                                                MEMLOAD / MEMRESET / MEMFORGET act on
MEMSNAP <path> <identity> [overwrite]           export the session's memory
MEMLOAD <path> <identity>                       import a snapshot
MEMRESET                                        zero the session's memory
MEMFORGET                                       drop the cached prompt state, keep the memory
```

Answers: `T <id>` / `DONE …` for a query (as `GEN`), `MEMCHUNK <done> <total>`
(one per prompt chunk of a commit: progress, so a long commit is not taken for a
silent engine), `MEMDONE <read> <prompt> <ms>` for a commit, `<CMD> OK` or
`ERR <why>`. The start line says
`READY <ctx> stop memt` for a MemT model. `MEM*` lines need an idle engine and
act on the engine's current session (`E.cur`): a one-session engine has only
that one; with `--batch N` the slots read their own memories and only an
explicit control line writes — `MEMSELECT <session>` picks the one the next
`MEMCOMMIT` / `MEMSNAP` / `MEMLOAD` / `MEMRESET` / `MEMFORGET` acts on (a
`MEMQUERY` is a request and picks its own session). `reset()` (a new conversation's
prompt) clears the GDN/KV state but **not** the memory: memory is the
conversation-independent store, and `MEMRESET` is the explicit clear.

`MEMFORGET` is the same conversation boundary without touching the memory: the
next query reads its whole prompt instead of reusing a shared prefix's KV and
GDN state. `memt_sglang`'s Nano configuration keeps no retained KV, so parity
(P6) and prompt-processing timings (P7) use `MEMFORGET` before every
independently measured request.

### Memory state through `terminal/serve/server.py`

Memory **reads** need nothing: the read is part of every forward, so a plain
`GEN` request already reads the session's state (a fresh state is zeros, which
add nothing). The engine advertises the memory in its start line —
`READY <ctx> stop memt` — and the server turns that into a write after every
**completed** turn:

- the request's reply must have stopped by itself: not `length` (cut by the
  cap), not `cancel` / `disconnect` / `error`, not trimmed by a `stop` string and
  not rewritten by the tool-markup retry (in both of those the tokens the engine
  wrote are not the text the next prompt will hold), and not ending in — or
  still able to become — a tool call: a tool chain's turns are one logical turn,
  written after its final answer;
- the server sends one `MEMCOMMIT` with the text the bank does not hold yet
  (the turn's prompt tail + its reply, template tokens included), while it still
  holds the engine's inference FIFO;
- a turn whose prompt does not begin with what the bank holds (another
  conversation, an edited history, a client that does not send the model's own
  text back) is not written; the bank is left intact and one line says so;
- an engine restart empties the bank, so the next completed turn writes its
  whole prompt and reply;
- `"memt_writes": false` in the run config keeps the reads and stops the writes.

So a chat through the app or any OpenAI-compatible client grows the memory as
it goes. The cost is one prefill of the turn's text after each reply (after a
restart, one prefill of the whole conversation) and the next request re-reading
its whole prompt: a commit drops the session's cached prompt state, because the
hidden states it used were computed against the pre-commit snapshot. A server started with
`"parallel": N` (one engine, several slots) runs requests concurrently, so it
does **not** write; each slot still reads its own session's memory. Replicas
(`"replicas": N`) are separate engine processes, each with a bank of its own, so
each writes its own while no other request runs on it. A split pair
(`split_serve`: a prefill engine and a decode engine) holds the layers in two
processes, so a commit would advance only the decode half: reads work, writes
are off (one line). Commits can also be
driven by hand through the line protocol
(`terminal/tools/memt_engine.py`, `MEMSELECT` for a batch engine's slot) or the
harnesses below.

The state itself — save it, load it, clear it — is `POST /v1/memt`, enabled by
`--memt-save-path DIR` (or `"memt_save_path"` in the run config); `status`
answers without the directory and reports `"memt": false` for a model without
memory:

```json
{"action": "status"}                              → {"memt", "loaded", "identity", "dir", "held_tokens", "sealed"}
{"action": "reset"}                               → MEMRESET; the bank is cleared
{"action": "save", "file": "chat.safetensors"}    → MEMSNAP; also writes chat.safetensors.json (held + sha256)
{"action": "save", "file": "…", "overwrite": true}   replace an existing file
{"action": "load", "file": "chat.safetensors"}    → MEMLOAD; the bank comes back from the sidecar
```

The identity is the GGUF's `memt.identity`: the converter writes it when the
checkpoint directory has the chat template and `tokenizer.json`, the engine
reports it as `INFO memt_identity=<hex>`, and the server passes it to
`MEMSNAP`/`MEMLOAD` — so a snapshot only loads back into a model with the same
identity (config, memory, codec, template, vocabulary; weights are not part of it). Saving also records the tokens the memory holds (the text of the turns
this server committed) in the JSON beside the snapshot; a snapshot loaded
without that sidecar still holds its tokens, but the server cannot name them,
so the bank is sealed and no later turn is written until a reset (a snapshot saved
from a sealed bank keeps the seal in its sidecar; the sidecar is bound to the
snapshot's SHA-256, so a copied or half-saved pair seals too). Refused: a model without memory, a
replica pool or split pair (each engine process has a memory of its own), a
`--parallel` engine (a snapshot needs the engine idle; `status` answers there), `save`/`load` without a directory, a file
name that is not one plain `.safetensors` name inside the directory (the
`/slots/0` rule), a symlink, or a file without an identity. Only from Strata's
own page, like `/v1/vram`.

`--batch N` and `--mtp N` are allowed with a MemT model: the memory is per
session, and a row window (a batch step, an MTP verify window) reads each row's
own session state — in one session the rows of a request read the same
pre-call snapshot, which is the documented semantics; with `--batch` each row
reads the state of its own slot. A `--batch` engine runs without CUDA graphs (a
captured graph would bake one session's state pointer); a one-session engine
keeps them, MTP's verify windows included. A memory control line needs the
engine idle.

`MEMRESET` / `MEMLOAD` / `MEMCOMMIT` are conversation boundaries: they drop the
session's cached prompt state (KV, GDN, checkpoints, pictures). The next request
reads its prompt afresh, because a commit's hidden states were produced against
the pre-commit snapshot and are not a query's prefix.

The identity is memt_sglang's snapshot identity (backbone config + `memt` object
+ `qwen-text-raw-v1` codec + chat template + vocabulary, SHA-256 of sorted JSON).
`terminal/tools/memt_engine.py` computes it from a checkpoint directory, drives
the control lines and uses `terminal/tools/memt_snapshot.py` for the file format.
The converter stores it in the GGUF (`memt.identity`, optional), the engine
reports it as `INFO memt_identity=<hex>`, and the server's `/v1/memt` route uses
it for `MEMSNAP`/`MEMLOAD`:

```python
with MemtEngine.spawn(["strata-q36", "--model", gguf, "--serve"]) as engine:
    engine.reset()
    engine.commit(document_ids)
    answer = engine.query(prompt_ids, max_new=64)
    engine.snapshot("/tmp/session.safetensors", identity)
```

Both codecs — `terminal/tools/memt_snapshot.py` and
`engine/include/strata/qwen35/memt_snapshot.hpp` — write the same bytes
(`memt_snapshot_test` pins a fixture produced by the Python one).

Mixture-of-Storage (the report's §2.3) is **not** implemented: the release ships
no code, config field or weights for it, so there is nothing to check a port
against. The audit refuses MoS-shaped configs by name; see §2.4 of
[MEMT-DESIGN.md](MEMT-DESIGN.md#24-mixture-of-storage-described-not-released-not-implemented).

## Tests (no GPU)

```sh
cd engine && cmake --preset cpu-check && cmake --build --preset cpu-check && ctest --preset cpu-check
python terminal/tools/test_memt_audit.py
python terminal/tools/test_memt_cpu_math.py
python terminal/tools/test_memt_to_gguf.py
python terminal/tools/test_memt_snapshot.py
python terminal/tools/test_memt_engine.py
python terminal/tools/test_memt_session_flow.py
python terminal/tools/test_memt_compare.py
python terminal/tools/test_memt_parity.py
python terminal/tools/test_memt_bench.py
python3.11 terminal/tools/test_memt_ref.py   # 3.11+: also validates against memt_sglang.protocol
python terminal/serve/test_memt_turns.py     # needs jinja2 + regex: the HTTP/server paths over a fake engine
```

What they cover: the real 753-tensor header audit (and a refusal of MoS-shaped
config fields), the FP32 scalar recurrence and its independent NumPy oracle
(`~1e-7`), the query/commit snapshot semantics, the CUDA kernels recompiled for
the CPU against the same reference (index math, strides, shared reductions), the
converter's every transform and the GGUF layout table, the snapshot codec on both
sides, the control-line client, the HTTP service, the server's per-turn memory
writes (`READY … memt`, one `MEMCOMMIT` per completed turn, the delta, the turns
it must not write, a refused and a silent engine), the memory state route
(`POST /v1/memt` status/reset/save/load, the sidecar prefix, the refusals),
the P6 procedure (against a stub
reference that writes real snapshots),
and the P7 measurement math. `test_memt_ref.py` additionally checks every
generation payload it builds with memt_sglang's own
`memt_sglang.protocol.validate_payload` when that checkout is importable (it
needs Python 3.11; `MEMT_SGLANG_PATH` names it), so wire-format drift fails on a CPU box rather than on the GPU
host.

## Precision and what a GPU comparison must measure

A self-contained English design and verification document — architecture,
semantics, formats, invariants, a performance model and the capability matrix —
is [docs/MEMT-DESIGN.md](MEMT-DESIGN.md).

`memt_sglang` stores and computes in BF16; this engine keeps the backbone in
Q8_0 (8-bit blocks) and the memory state in FP32, and runs the read/update in
FP32 (the reference's `tl.dot` runs in BF16). Bitwise equality is therefore not
the bar. A GPU run must report, on the same checkpoint and prompts:

- logits/token agreement on short prompts (greedy, and the top-k overlap);
- the FP32 memory state after identical commits;
- `memt-memory-v1` round-trip HF → engine → HF;
- tok/s, TTFT and TPOT against memt_sglang on the same card, three runs, median.

Two harnesses do that; both write a JSON report next to the artifacts, and both
treat a missing server as an error, never as a pass.

**P6 — the memory state and the generated tokens.** Start the reference first
(`sh start_memt_nano.sh /models/MemT-1.0-Nano --state-directory /data/memt-state`),
then:

```sh
python terminal/tools/memt_parity.py \
    --ref-url http://127.0.0.1:30000 --ref-state-dir /data/memt-state \
    --engine ./build/cuda/strata-q36 \
    --engine-arg=--model --engine-arg=/models/MemT-1.0-Nano.memt.gguf --engine-arg=--serve \
    --engine-model-dir /models/MemT-1.0-Nano \
    --commit evidence.ids --prompt q1.ids --prompt q2.ids \
    --outdir runs/p6 --max-new 32 --min-cosine 0.999 --ready-timeout 1200
```

`--engine-arg=` needs the equals form: argparse takes a value that starts with
`--` as another option, so `--engine-arg --model` fails with "expected one
argument". `--serve` is part of the engine's args because only serve mode prints
`READY`; without it the engine runs as a `--tokens` invocation and exits.
`--ready-timeout` bounds the reference's readiness (the engine is waited on
without a timeout), and its 30 s default is shorter than a BF16 load.

`evidence.ids`, `q1.ids`, … are token-id files (JSON list, `{"ids": [...]}`, or
comma/space separated). The procedure: the reference commits from an empty
memory and exports; the engine commits the same tokens from an empty memory and
exports; the two files are compared with `memt_compare.py`; both answer the same
prompts and the ids are compared; then the engine's snapshot is copied into the
reference's state directory and imported there; the prompts are answered again
through it and the reference's memory is exported back and compared with our
file — that last step is what proves the HF side really reads, and reproduces, a
C++-written file. The identity comes from the reference's
`GET /memt/capabilities` (`snapshot_identity`), so the two sides cannot silently
disagree about which model the state belongs to; `--engine-model-dir` recomputes
that identity from the checkpoint the GGUF was converted from and fails before
committing anything if it differs. Both sides generate greedily
(`temperature` 0 — the reference's default is not greedy). Exit status: 0 pass,
1 a comparison failed, 2 setup/transport error.

The token bar is exact equality by default; the state bar is `--min-cosine`
(0.999) **and** `--max-rel-l2` (0.25, i.e. `||ref - engine|| / ||ref||`), because
cosine is scale-invariant and would accept a state that is uniformly too large.
Both are gross-error ceilings, not the agreed tolerance. Q8_0 versus BF16 may
drift on a long generation; the report always carries the common-prefix length
and every layer's `rel_l2`, so the tolerance is agreed from those measured
numbers (and then passed as `--allow-token-drift` / a wider `--max-rel-l2`), not
chosen before the run.

**P7 — the speed.** Same card, same prompt ids, same `--max-new`, greedy, a
discarded warmup, then `--runs` measured requests:

```sh
python terminal/tools/memt_bench.py --side both \
    --ref-url http://127.0.0.1:30000 \
    --engine ./build/cuda/strata-q36 \
    --engine-arg=--model --engine-arg=/models/MemT-1.0-Nano.memt.gguf --engine-arg=--serve \
    --prompt q1.ids --max-new 128 --runs 3 --outdir runs/p7 --ready-timeout 1200
```

Each run starts from an empty prompt cache: the engine is sent `MEMFORGET` (so a
prefix shared with an earlier request is not reused) and the reference's Nano
configuration keeps no retained KV. The report carries the median TTFT, TPOT,
end-to-end latency and tok/s per side, the engine/reference ratio, the GPU
(`nvidia-smi`), and the exact condition (argv, sampling, run count). These are
client-visible numbers: the reference is measured through its HTTP/SSE
`/generate`, the engine through its line protocol, so each side carries its own
transport's overhead — the same way an application would see it. TPOT is
derived from the request's end time — the reference streams SSE frames, so if it
batches them its TTFT is late and its TPOT a lower bound, which the report notes
(`frames < output_tokens - 1`); tok/s and end-to-end latency are unaffected.
If the two sides generated different numbers of tokens, the row is not a
comparison at all: it is marked `NOT COMPARABLE`, no ratio is printed, and the
run exits non-zero.

Until those runs exist, treat parity and speed as unmeasured.
