# MemT-1.0-Nano on a GPU host: runbook

Status: **written from the source and the CPU test suite on 2026-11-26. No step below has run on a GPU yet.** No CUDA
build, no real checkpoint conversion, no reference server, no P6 / P7 report exists. Every flag, file and message in
this page was checked against the code; the VRAM sizes and timings are expectations from the code's own arithmetic,
not measurements. When the first GPU run happens, fill §1.1 with what the machine printed and mark each step in §3
that passed.

The code is on `feat/memt-1.0-nano`. This page is on `docs/memt-gpu-runbook` until someone merges it; if the file is
not in your checkout, copy it there or check out both branches.

See [MEMT.md](MEMT.md) for what the memory block is and [MEMT-DESIGN.md](MEMT-DESIGN.md) for the design.

## 0. The three judgment points

| # | Question | Where the answer is | Bar |
| --- | --- | --- | --- |
| J1 | Does the engine load the converted file and run? | the engine's own stderr lines (§4.1) | `READY … stop memt`, a non-empty `INFO memt_identity`, no refusal, a sensible `gen:` answer |
| J2 | Does the memory match the reference? | `runs/p6/parity.json` | snapshot cosine ≥ 0.999 and `rel_l2` ≤ 0.25 per layer, tokens identical, roundtrip PASS |
| J3 | Is the speed measurable? | `runs/p7/bench.json` | medians for both sides and a printed ratio, or an explicit `NOT COMPARABLE` with the reason |

A missing server, a refused load or an unrun comparison is **not** a pass. `memt_parity` and `memt_bench` exit non-zero
and say why; treat that as the result.

## 1. Machine

### 1.1 Collect the facts first

Run these on the GPU host and paste the output into the table below. Do not guess the values.

```sh
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi -L
nvcc --version
free -g; nproc
df -h "$MEMT_ROOT"        # MEMT_ROOT: the large disk you will use (§2)
uname -m; python3 --version; cmake --version; ninja --version; g++ --version
```

| Fact | This machine |
| --- | --- |
| GPU(s): model, count, VRAM each | |
| Free VRAM per card with nothing running (`nvidia-smi`) | |
| Host RAM (total / available) | |
| Driver (`nvidia-smi` top right) | |
| CUDA toolkit (`nvcc --version`) | |
| Free disk on `$MEMT_ROOT` | |
| CPU cores, OS, gcc, cmake, ninja, python | |

### 1.2 What the work needs

| Need | Why | Minimum |
| --- | --- | --- |
| Linux x86-64 | the reference server (`memt_sglang`) is Linux + CUDA only | — |
| One NVIDIA card, sm_80 or newer | the CUDA kernels need tf32 / bf16 math | sm_89 = RTX 40 |
| Driver ≥ the CUDA toolkit you build with | `nvcc` 12.9 or 13.x both work | `nvidia-smi`'s "CUDA Version" ≥ `nvcc`'s |
| Host RAM | the hybrid mode pins **every routed expert in host memory** (~34 GB) | 64 GB, 128 GB comfortable |
| Disk | BF16 checkpoint ~70 GB + tokenizer GGUF ~13–20 GB + converted GGUF ~37 GB + runs | 150 GB free, 200 GB comfortable |
| Toolchain | CMake ≥ 3.24, Ninja, gcc ≥ 11 (C++20), Python ≥ 3.11 | — |

### 1.3 Can this machine run the reference at all?

`MemT-1.0-Nano` is served **unquantized BF16** — about 70 GB of weights. The launcher's example uses `--tp-size 2`.
A 24 GB card cannot hold its share; **no 1- or 2-card 4090 box can run the reference**. Decide which path you are on
before building anything:

| Cards on the box | Reference | Engine | What to do |
| --- | --- | --- | --- |
| 1×24 GB | no | yes, hybrid (§5) | engine-only steps (§3.1–§3.6); record that J2/J3 are blocked by hardware |
| 2×24 GB | no (70 GB > 48 GB) | yes | same |
| 4×24 GB | possible with `--tp-size 4`, `--mem-fraction-static` ~0.9 | needs a free card | 5th card or a second host, otherwise run the two sides one after the other and accept that P7 is not a same-card comparison |
| 8×24 GB | yes (`--tp-size 4`, cards 4–7) | yes (card 0) | the full path; P7 compares the same card model on separate physical cards — say so in the report |
| ≥2 cards with ≥48 GB free each | yes (`--tp-size 2`, the default) | yes | the full path; the reference and the engine can share the box if one card is kept for the engine |

The reference has no quantized mode (`quantization` must stay `None`); there is no "run the reference on the 4090" option.

If J2/J3 are blocked by card count, still do §3.1–§3.6 and write the blocker into the issue: that is a real result, not
a failure of the engine.

## 2. Where things go

Pick one root and use it everywhere. Nothing is installed under `/models` unless you choose that.

```sh
export MEMT_ROOT=/data/memt          # a large local disk; every path below is under it
export MEMT_CODE=$MEMT_ROOT/code
export MEMT_MODELS=$MEMT_ROOT/models
export MEMT_STATE=$MEMT_ROOT/state
export MEMT_RUNS=$MEMT_ROOT/runs

mkdir -p "$MEMT_CODE" "$MEMT_MODELS" "$MEMT_STATE" "$MEMT_RUNS"
```

| What | Path | Size |
| --- | --- | --- |
| Fork checkout (`feat/memt-1.0-nano`) | `$MEMT_CODE/strata-memt` | code |
| Reference checkout (`MemT-1.0/inference`) | `$MEMT_CODE/MemT-1.0` | code + a Python env |
| HF checkpoint (BF16, all shards + `tokenizer.json` + chat template) | `$MEMT_MODELS/MemT-1.0-Nano/` | ~70 GB |
| A qwen35moe GGUF, used only for its tokenizer | `$MEMT_MODELS/Qwen3.6-35B-A3B-tokenizer.gguf` | 13–20 GB |
| Converted model the engine reads | `$MEMT_MODELS/MemT-1.0-Nano.memt.gguf` | ~37 GB |
| Reference's snapshot directory (`--state-directory`) | `$MEMT_STATE/` | 40 MB per snapshot |
| P6 / P7 reports and snapshots | `$MEMT_RUNS/p6`, `$MEMT_RUNS/p7` | small |

The checkpoint comes from <https://huggingface.co/MemTensor/MemT-1.0-Nano>. Any Unsloth Qwen3.6-35B-A3B quant works
for `--tokenizer-from` (only its `tokenizer.*` fields are copied), but its vocabulary must be 248320 tokens.

## 3. Commands, in order

Run everything from `$MEMT_CODE/strata-memt` unless a step says otherwise.

### 3.1 Get the code and a Python environment

```sh
git clone <your fork> "$MEMT_CODE/strata-memt" && cd "$MEMT_CODE/strata-memt"
git switch feat/memt-1.0-nano          # the branch that has the MemT work
python3 -m venv .venv
.venv/bin/pip install -r terminal/requirements.txt   # numpy, jinja2, regex, pyyaml
```

The engine itself needs no Python at runtime. The harnesses (`terminal/tools/memt_*.py`) need this venv.

### 3.2 CPU tests (no GPU, ~1 min)

```sh
cd engine
cmake --preset cpu-check
cmake --build --preset cpu-check
ctest --preset cpu-check
```

Expect `100% tests passed` (178 tests on the Mac this page was written on; this preset marks four tests "not run":
`io_priority_test`, which needs a cgroup, and three expert/pool tests that need native experts). The five MemT tests
must be among them:

```sh
ctest --preset cpu-check -R memt
```

Then, back at the repository root, the Python side:

```sh
cd "$MEMT_CODE/strata-memt"
for t in test_memt_audit test_memt_cpu_math test_memt_to_gguf test_memt_snapshot \
         test_memt_engine test_memt_session_flow test_memt_compare test_memt_parity test_memt_bench; do
  .venv/bin/python "terminal/tools/$t.py" || echo "FAILED: $t"
done
.venv/bin/python terminal/serve/test_memt_turns.py     # ~5 min: it starts fake servers
```

All ten pass on the CPU-only machine this page was written on. If one fails here, stop: a GPU run will not fix it.

### 3.3 CUDA build for sm_89

The source directory is `engine`, not the repository root, and the default architecture is sm_120 (RTX 50 only). Both
matter — a build without `89` compiles but cannot run on a 4090.

```sh
cd "$MEMT_CODE/strata-memt"
cmake -S engine -B build/cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/cuda --target strata-q36 -j "$(nproc)"
```

The CUDA build needs the CPU expert pool (the default). Do not pass `-DSTRATA_NATIVE_EXPERTS=OFF`: the 24 GB card
needs the hybrid mode, and a build without the pool refuses to start ("this build has no CPU expert pool").

Check the binary really has sm_89 code:

```sh
cuobjdump --list-elf build/cuda/strata-q36 | grep -c sm_89    # > 0
```

The binary is `build/cuda/strata-q36`. A released Strata engine is not a substitute: it does not know the
`blk.<i>.memt.*` tensors.

### 3.4 Convert the checkpoint

Look before converting (the audit needs only `config.json` and the shard index):

```sh
.venv/bin/python terminal/tools/memt_audit.py \
  --config "$MEMT_MODELS/MemT-1.0-Nano/config.json" \
  --index  "$MEMT_MODELS/MemT-1.0-Nano/model.safetensors.index.json"
```

The audit prints the tensor mapping and refuses Mixture-of-Storage-shaped configs. Then convert:

```sh
.venv/bin/python terminal/tools/memt_to_gguf.py \
  --checkpoint "$MEMT_MODELS/MemT-1.0-Nano" \
  --tokenizer-from "$MEMT_MODELS/Qwen3.6-35B-A3B-tokenizer.gguf" \
  --out "$MEMT_MODELS/MemT-1.0-Nano.memt.gguf"
```

The converter streams one tensor at a time (a 36 B-parameter checkpoint is never held in RAM at once), writes a temp
file next to the output and renames it at the end. It prints a JSON summary: `tensors`, `bytes`, `memory_layers`
(should be `3, 7, 11, …, 39`) and `identity`. **Save that JSON** to `$MEMT_RUNS/convert.json`.

Warnings to understand:

* `memt.identity not written: …` — the checkpoint directory lacks `tokenizer.json` or the chat template. The engine
  still runs, but `MEMSNAP` / `MEMLOAD` and `POST /v1/memt` save/load will refuse. Fix the checkpoint directory and
  reconvert before P6.
* `… has N tokens, the checkpoint's vocab is 248320` — wrong `--tokenizer-from` file. Pick another qwen35moe GGUF.

### 3.5 First load, no reference (J1, part 1)

```sh
./build/cuda/strata-q36 \
  --model "$MEMT_MODELS/MemT-1.0-Nano.memt.gguf" \
  --max-context 8192 \
  --chunk 1024 \
  --kv fp16 \
  --expert-vram 16 \
  --cpu-threads "$(nproc)" \
  --tokens 1,2,3 --n 4          # any ids; 1,2,3 are only a smoke prompt
```

What the run should print (stderr unless noted):

```
strata-q36: …: N tensors (layout checked), D GiB on the GPU in … s (the routed experts are placed next)
strata-q36: MemT-1.0 memory: 10 layers x 4.00 MiB per session (FP32 state), the same again once a commit has run …
strata-q36: hybrid experts: S of 256 per layer in VRAM (A GiB), all B GiB in pinned host memory, …
gen: …                                    (stdout)
```

`D` is the dense part only — embeddings, attention, norms — a few GiB, not the ~37 GB file. If the next line is
`every routed expert in VRAM (…)`, the card had room for all experts; fine, but a 24 GB card normally will not.

### 3.6 Start the reference

The reference must run before P6 / P7. Pick card IDs it may use and keep at least one card for the engine.

```sh
cd "$MEMT_CODE/MemT-1.0/inference"
uv venv .venv --python 3.11 && . .venv/bin/activate
uv pip install '.[server]'        # torch 2.11, sglang 0.5.16, flashinfer cu13, transformers 5.12.1

export CUDA_VISIBLE_DEVICES=4,5,6,7       # the cards the reference may use (example: an 8×4090 box)
export MEMT_PYTHON="$PWD/.venv/bin/python"
export MEMT_PORT=30000
export MEMT_STATE_DIRECTORY="$MEMT_STATE"
sh start_memt_nano.sh "$MEMT_MODELS/MemT-1.0-Nano" \
  --tp-size 4 --port 30000 --state-directory "$MEMT_STATE" \
  --mem-fraction-static 0.88
```

Wait until it answers, then record the identity:

```sh
curl -s http://127.0.0.1:30000/memt/capabilities | python3 -m json.tool
```

`snapshot_identity` is the value the engine must print and the harness must see; `chunk_tokens` should be 2048. A
server that is still loading prints nothing here — a 70 GB BF16 load takes minutes, not seconds.

The `--mem-fraction-static` is the only number to tune on this line: on four 24 GB cards 0.88 leaves little room
above 70 GB of weights, so a startup OOM is usually solved by lowering it (0.85, then 0.80) or by spreading the
reference over eight cards with `--tp-size 8`. On two 48 GB cards the script's own 0.60 is the sane default; drop the
`--tp-size` / `--mem-fraction-static` overrides there.

### 3.7 Build the token-id files

P6 and P7 take token ids, not text, so the two sides cannot disagree about tokenization. Use the checkpoint's own
tokenizer:

```sh
cd "$MEMT_CODE/MemT-1.0/inference"
.venv/bin/python - "$MEMT_MODELS/MemT-1.0-Nano" "$MEMT_RUNS" <<'PY'
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
tok = AutoTokenizer.from_pretrained(str(ckpt))
texts = {
    "evidence": "<the document you want the memory to hold>",
    "q1": "<a question about that document>",
    "q2": "<a second question>",
}
for name, text in texts.items():
    ids = tok(text, add_special_tokens=False)["input_ids"]
    (out / (name + ".ids")).write_text(json.dumps(ids))
    print(name, len(ids))
PY
```

Keep the commit shorter than 2048 tokens: at or above the reference's `chunk_tokens` the reference splits the write
into several events and the cleaner comparison is lost (the harness notes it, it does not fail).

### 3.8 P6 — memory state and generated tokens

```sh
cd "$MEMT_CODE/strata-memt"
CUDA_VISIBLE_DEVICES=0 timeout 2400 .venv/bin/python terminal/tools/memt_parity.py \
  --ref-url http://127.0.0.1:30000 \
  --ref-state-dir "$MEMT_STATE" \
  --engine ./build/cuda/strata-q36 \
  --engine-arg=--model --engine-arg="$MEMT_MODELS/MemT-1.0-Nano.memt.gguf" \
  --engine-arg=--serve --engine-arg=--max-context --engine-arg=8192 \
  --engine-arg=--kv --engine-arg=fp16 --engine-arg=--expert-vram --engine-arg=16 \
  --engine-arg=--cpu-threads --engine-arg="$(nproc)" \
  --engine-model-dir "$MEMT_MODELS/MemT-1.0-Nano" \
  --commit "$MEMT_RUNS/evidence.ids" \
  --prompt "$MEMT_RUNS/q1.ids" --prompt "$MEMT_RUNS/q2.ids" \
  --outdir "$MEMT_RUNS/p6" --max-new 32 --ready-timeout 1200
```

Five traps are in that command on purpose:

1. `--engine-arg=--model` — with the space form (`--engine-arg --model`) argparse says "expected one argument", because
   the value starts with `--`. Always use `=`.
2. `--engine-arg=--serve` — the line protocol and `READY` only exist in serve mode. Without it the engine treats the
   args as a `--tokens` run and exits before `READY`.
3. `--ready-timeout 1200` — this bounds the **reference's** readiness, not the engine's (the engine is waited on
   without a timeout). The 30 s default fails on a BF16 load.
4. `CUDA_VISIBLE_DEVICES=0` — pins the engine to card 0; the harness inherits the environment. The engine's own
   `--gpu` flag is applied by the `strata` wrapper, not by the standalone binary.
5. `timeout 2400` — the harness waits on the engine without any timeout of its own; this keeps a hung load from
   stalling the whole session.

The engine args after `--serve` are the **24 GB recipe** (`--kv fp16`, `--expert-vram 16`). On a card with room for all
experts, drop `--expert-vram` (it forces the hybrid mode when the experts would otherwise fit) and keep `--kv fp32`
unless you need the room.

`--ref-state-dir` must be a local path this process can read and write: the harness reads the reference's exported
file there and copies the engine's snapshot into it for the roundtrip. If the reference is on another host, mount its
state directory over NFS/sshfs first; a remote `--ref-state-dir` without a mount fails at the first comparison.

### 3.9 P7 — the speed

```sh
cd "$MEMT_CODE/strata-memt"
CUDA_VISIBLE_DEVICES=0 timeout 2400 .venv/bin/python terminal/tools/memt_bench.py --side both \
  --ref-url http://127.0.0.1:30000 \
  --engine ./build/cuda/strata-q36 \
  --engine-arg=--model --engine-arg="$MEMT_MODELS/MemT-1.0-Nano.memt.gguf" \
  --engine-arg=--serve --engine-arg=--max-context --engine-arg=8192 \
  --engine-arg=--kv --engine-arg=fp16 --engine-arg=--expert-vram --engine-arg=16 \
  --engine-arg=--cpu-threads --engine-arg="$(nproc)" \
  --prompt "$MEMT_RUNS/q1.ids" --max-new 128 --runs 3 --ready-timeout 1200 \
  --outdir "$MEMT_RUNS/p7"
```

`--ready-timeout` here bounds both the reference and the engine; the 30 s default is far too short. Same note as P6:
drop `--expert-vram` and `--kv fp16` on a card that holds every expert.

## 4. The three judgment points in detail

### 4.1 J1 — did it load and run?

Good signs, in order:

| Line | Means |
| --- | --- |
| `strata-q36: … N tensors (layout checked), X GiB on the GPU` | the GGUF header and tensor layout were accepted |
| `strata-q36: MemT-1.0 memory: 10 layers x 4.00 MiB per session …` | the memory block is in the file and its per-session state was allocated |
| `strata-q36: every routed expert in VRAM (X GiB)` or `hybrid experts: S of 256 per layer in VRAM (A GiB), all B GiB in pinned host memory` | expert placement succeeded; S and B are what §5 tunes |
| `INFO memt_identity=<64 hex>` (serve only) | the converter wrote an identity; this is what P6 compares |
| `READY <ctx> stop memt` (serve only) | the engine is serving and knows it has memory |
| `gen: 123 456 …` (stdout, `--tokens` mode) | a forward pass and greedy decode completed |

Bad signs and where to go:

| What you see | Cause | Action |
| --- | --- | --- |
| `strata-q36: <path> is <type>, which the native MMVQ does not take` | not the converter's Q8_0/F32 file | reconvert; do not hand-quantize |
| `strata-q36: missing blk.<i>.memt.…` | a file that is not the MemT GGUF | use `MemT-1.0-Nano.memt.gguf` |
| `head_dim` / `state` / `conv` / `experts` refusal | a different model's GGUF | wrong checkpoint |
| `this build has no CPU expert pool` | built with `STRATA_NATIVE_EXPERTS=OFF` | rebuild (default is ON) |
| `out of memory: … needs X MiB, Y MiB free` | buffers, KV or slots do not fit | §5, in the order given |
| `a context of N tokens does not fit on this graphics card … M tokens fit here` | the single session's K/V and state alone are too big | `--max-context` to `M` (or lower) |
| `the engine did not print READY within … s` | harness timeout, not a load failure | raise `--ready-timeout`; `--no-warmup` shortens startup |
| no `INFO memt_identity` but otherwise running | converter could not compute the identity | add `tokenizer.json` + chat template to the checkpoint dir and reconvert |

A `--batch` above 1 on a MemT file is allowed, but it turns CUDA graphs off (the memory read follows the row's
session); use `--batch 1` for the standard P6/P7 runs.

### 4.2 J2 — P6 pass or fail

Read `runs/p6/parity.json` (and the human summary `memt_parity` prints). The exit code is the verdict: 0 pass, 1 a
comparison failed, 2 setup/transport error (nothing was compared).

| Field | What it proves | Default bar |
| --- | --- | --- |
| `comparisons.snapshot.layers.<i>.cosine`, `.rel_l2`, `.max_abs` | the FP32 memory after identical commits | cosine ≥ `--min-cosine` 0.999 **and** `rel_l2` ≤ `--max-rel-l2` 0.25 |
| `comparisons.tokens.rows.<name>.identical_prefix` | the generated ids | exact equality unless `--allow-token-drift` was agreed |
| `comparisons.roundtrip` | the reference loaded and ran on our C++-written file | PASS |
| `comparisons.roundtrip_state` | the reference's memory, exported again, equals ours | same state bar |

How to read a failure:

* **Exit 2, `the reference server is not up`** — start the reference (§3.6) and wait for `/memt/capabilities`.
* **Exit 2, `the engine checkpoint's snapshot identity is X, the reference's is Y`** — the GGUF was converted from a
  different checkpoint or with a different tokenizer; reconvert from the directory the reference serves.
* **Exit 2, `the reference exported … but <path> does not exist`** — `--ref-state-dir` is not the server's
  `--state-directory`, or the mount is not shared.
* **One layer low, others fine** — most likely that memory layer's indexing; compare `max_abs` and the layer number
  against the commit length. Not a tolerance question.
* **All layers uniformly low** — commit semantics or scale (wrong prompt ids? different `chunk_tokens`? the commit
  split?), not per-layer math.
* **Tokens diverge after k** — Q8_0 against BF16 drift is expected on a long generation. The report carries the
  common prefix; agree a tolerance from it, then rerun with `--allow-token-drift` (or a wider `--max-rel-l2`) **after**
  the agreement, never to make the first run green.
* **`roundtrip` PASS but `roundtrip_state` FAIL** — the reference read the file but wrote a different state back;
  suspect the file's dtype/order, not the tokens.

The two defaults are gross-error ceilings, not the agreed tolerance. The real numbers are the measured `rel_l2` per
layer.

### 4.3 J3 — P7 produced numbers

`runs/p7/bench.json` must contain, per prompt and per side, the median TTFT, TPOT, end-to-end latency and tok/s, the
ratio, the GPU (`nvidia-smi`) and the exact argv and sampling. "Produced numbers" means the row is comparable:

| In the report | Meaning | Action |
| --- | --- | --- |
| `engine_vs_reference` with `tps`, `ttft_ms`, `tpot_ms` | a ratio was printed (>1 means the engine is faster: `tps` is engine/reference, the ms fields are reference/engine) | done; quote the medians, not one run |
| `NOT COMPARABLE (different output token counts)` | the two sides did different work | not a P7 result; fix the prompt/`--max-new` and rerun |
| `frames < output_tokens - 1` note | the reference batched SSE frames; its TTFT is late and TPOT a lower bound | tok/s and end-to-end are still valid; say so in the report |
| No warmup / fewer than 3 runs | noise | the defaults (1 discarded warmup, 3 runs, median) are the minimum |

P7 is a measurement, not a pass/fail: the deliverable is the JSON plus a sentence that says whether the engine is
faster, slower or not comparable, and why. With the hybrid mode on a 24 GB card the CPU share is the dominant cost;
that number is not a claim about how a 4090 "should" run this model.

## 5. When the card is too small (RTX 4090, 24 GB)

The routed experts are the problem: 256 experts × 40 layers at Q8_0 are ~34 GB, against ~24 GB of card. The engine's
hybrid mode keeps every expert in host memory and only S of 256 per layer on the card.

The arithmetic the engine prints at startup is the one to use:

```
hybrid experts: S of 256 per layer in VRAM (A GiB), all B GiB in pinned host memory, …
```

`S = floor((expert budget − staging) / per_slot)`, `per_slot` ≈ ~34 GB / 256 ≈ 134 MB, staging is two buffers of the
largest layer's experts (~1.7 GB for Q8_0). With `--expert-vram 16` expect roughly 100 experts per layer (~40 %) and
~34 GB pinned in host memory.

Knobs, in the order that keeps the run honest:

| Order | Flag | Effect | Notes |
| --- | --- | --- | --- |
| 1 | `--expert-vram GiB` | the experts' total VRAM budget (slots + staging) | the main dial. Start at 16 on a 24 GB card; drop by 2 when the plan refuses, raise by 1 while it still starts |
| 2 | `--expert-slots N` | forces S per layer instead of deriving it | use after reading the startup line, when you want room left for a longer context |
| 3 | `--kv fp16` | halves the K/V cache | small on an 8K context (~0.2 GB), large at 32K; record it, it is part of the condition |
| 4 | `--max-context N` | shrinks the K/V and the attention score buffer | P6/P7 prompts are short; 8192 is enough. Raise only when the commit needs it |
| 5 | `--chunk N` (or `--prefill N`) | smaller prompt buffers (~0.5 MiB per token) | 1024 instead of the auto 2048 saves ~0.5 GB; larger chunks give faster prompts when they fit |
| 6 | `--cpu-threads N` | threads of the CPU expert pool | `$(nproc)` minus 1–2; too many contenders hurt the GPU side |
| 7 | `--hybrid-rows N` | rows a batch runs on slots + pool together (default 128) | leave it unless the CPU share starves; it does not change S |
| 8 | `--peer-device N` | a **second** card's tier for more experts | needs a free card; `--expert-profile` ranks the hot experts onto it; useless when the reference owns that card |
| 9 | `STRATA_Q36_PLACE_DEBUG=1` | prints the chunk rule and the exact GiB the decision used | set it on the first run; it explains every refusal |
| 10 | `STRATA_Q36_CHUNK=1024` | forces a chunk after the auto rule | same effect as `--chunk`, for when you want the auto rule to still pick the rest |

**`--expert-profile` does not help a single card.** Card 0's hybrid slots hold experts `0 … S-1` by index, not the
hottest ones; the ranked profile is only consumed by `--peer-device` (the second card's tier). Do not spend time on
`--dump-expert-profile` unless a second card is in play.

Starting point for one 4090, 8K context:

```sh
STRATA_Q36_PLACE_DEBUG=1 CUDA_VISIBLE_DEVICES=0 ./build/cuda/strata-q36 \
  --model "$MEMT_MODELS/MemT-1.0-Nano.memt.gguf" --serve \
  --max-context 8192 --chunk 1024 --kv fp16 \
  --expert-vram 16 --cpu-threads "$(nproc)"
```

Read the startup lines, then adjust **one** knob at a time. If `place_experts` refuses with "not enough VRAM left for
the hybrid experts' buffers and slots", lower `--expert-vram`; if a session's context is refused, lower
`--max-context`; if a staging allocation fails, lower `--chunk` (the engine falls back to a non-streamed prompt path
and says so, but prompts get slower).

Expect the hybrid decode to be CPU-bound: most of a token's routed experts run on the host. Treat 24 GB P7 numbers as
"this is what a 4090 with 40 % of the experts on the card does", not as the model's speed.

Do not use `--kv fp8` / `--kv fp4` for a first P6: they change the numerics under measurement. If they are the only
way to fit, record the flag in the report and re-negotiate the tolerance from the measured numbers.

Host memory is a hard prerequisite: the hybrid arena is `cudaHostAlloc`-ed (pinned) for all experts, ~34 GB. On a
64 GB box leave the rest for page cache; a 32 GB box will fall back to pageable memory (slower) or fail outright.

## 6. Failure catalog

| Symptom | Most likely cause | Fix |
| --- | --- | --- |
| `no kernel image is available for execution on the device` | built for sm_120 (the default) | rebuild with `-DCMAKE_CUDA_ARCHITECTURES=89` |
| `cmake` error `source directory … does not contain CMakeLists.txt` | ran `-S .` at the repository root | `-S engine` |
| `argument --engine-arg: expected one argument` | `--engine-arg --model` | `--engine-arg=--model` |
| `the engine exited before READY` | engine args lack `--serve` | add `--engine-arg=--serve` |
| `the engine did not print READY within 30.0 s` | P7's default timeout | `--ready-timeout 1200` |
| `the reference server is not up` | server still loading, or wrong port | wait for `/memt/capabilities`; raise `--ready-timeout` |
| `identity … does not match` | GGUF from a different checkpoint/tokenizer | reconvert from the served checkpoint |
| snapshot comparison fails on every layer | commit ids are not the same tokens | rebuild `.ids` files from the checkpoint tokenizer |
| `the reference exported … but … does not exist` | `--ref-state-dir` mismatch / no shared mount | point it at the server's `--state-directory` |
| P7 `NOT COMPARABLE` | different output token counts | shorten/adjust the prompt or accept it is not a comparison |
| engine takes minutes to reach `READY` | 37 GB GGUF load + 34 GB host arena + slot copies | normal; measure it, do not time out |

### 6.1 What this page does not cover

* The HTTP server (`strata serve`, `POST /v1/memt`, per-turn writes): see [MEMT.md](MEMT.md) §"Memory state through
  `terminal/serve/server.py`". P6/P7 do not need it.
* The `--batch 2` two-conversation P6 variant (the design asks for it once per session mode). The parity harness
  drives one session; the two-session claim needs a manual `MEMSELECT` run and is not automated here.
* Vision, MTP drafts (`--mtp`), `--peer-device` tuning and Mixture-of-Storage: out of scope for this model's first run.

## 7. What is verified, as of 2026-11-26

| Claim | Evidence |
| --- | --- |
| The CPU test suite passes (C++: 178 registered, 174 run, 4 skipped by the preset; Python: 10 files) | run on a CPU-only Mac, 2026-11-26 |
| The converter, snapshot codec, control client, P6/P7 math exist and are tested against stubs | `test_memt_*` in `terminal/tools`, `terminal/serve` |
| The CUDA build, the real checkpoint conversion, the reference server, P6 and P7 | **not run anywhere yet** |

When the first GPU run finishes, update this table and §1.1 with the machine it ran on. Until then, treat every number
above as an expectation.
