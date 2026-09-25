# ASCII Pong v2 — LLM Opponent Architecture

**Date:** 2026-09-25
**Status:** Proposed
**Base:** `ascii_pong.py` (v1, deterministic AI)

## 1. Problem framing

Replace the deterministic right-paddle controller with an LLM served by
Ollama (remote/cloud box, same REST API as the local benchmark at
`localhost:11434`). The LLM must, **within the ball's flight time**, decide
(a) where to intercept and (b) how to strike (offset → spin).

Hard constraints:

- **Latency budget:** field is ~58 cells wide, ball flies at 18–55 cells/s
  → **1.0–3.2s** of flight time after the player's hit. Model must answer
  in well under 1s to leave the paddle time to travel.
- **Capability ceiling (from GEMMA3-270M-STUDY.md):** the model is strong
  at *semi-structured → JSON extraction*, weak at *reasoning/math/
  multi-class judgment*. Predicting a wall-bouncing trajectory **is math** —
  a 270M model will fumble it.

## 2. Component breakdown

```
ascii_pong.py (v1, untouched)
        │ copy/extend
        ▼
ascii_pong_llm.py (v2, single file, stdlib-only)
┌────────────────────────────────────────────────────────┐
│ Game loop (curses, 80ms tick) — never blocks on LLM    │
│   on player hit ('paddle', vx>0):                      │
│     1. TrajectoryPredictor ── deterministic: clones    │
│        ball state, steps a copy forward → intercept_y, │
│        t_arrive (microseconds, exact)                  │
│     2. AsyncAdvisor.submit(state) ── daemon thread,    │
│        HTTP POST /api/generate, timeout 1.0s           │
│     3. FallbackController ── meanwhile moves paddle    │
│        toward intercept_y (today's AI = safety net)    │
│     4. on response: parse → validate → clamp →         │
│        PaddleController retargets                      │
│        target = intercept_y − hit_offset·PADDLE_H/2    │
│        (existing Ball.bounce turns offset into spin)  │
└────────────────────────────────────────────────────────┘
```

Modules inside the file:

| Module | Responsibility |
|---|---|
| `TrajectoryPredictor` | Deterministic forward simulation → intercept point + arrival time |
| `PromptBuilder` | Few-shot prompt, `key=value` state encoding |
| `AsyncAdvisor` | Daemon thread, HTTP POST, 1.0s timeout, last-wins result slot |
| `ResponseParser` | Brace-extraction, clamp, validate |
| `PaddleController` | Blends fallback target ↔ LLM target |
| `CircuitBreaker` | Disables LLM after 3 consecutive failures |

## 3. Decision rationale

- **Hybrid intelligence (the core decision):** physics computes *where*,
  LLM decides *how*. This routes each subtask to the component the study
  proved competent. The LLM's job becomes "pick an offset in [-1, 1] + a
  target nudge" — a tiny, few-shot-anchored choice, not ballistic math.
  The LLM still *owns* the strategic decision (spin direction, safe vs
  aggressive), and its errors become human-like misses — which is the fun.
- **Async, last-wins:** the curses loop is single-threaded at 80ms ticks;
  blocking HTTP would freeze rendering. One daemon thread, one result
  slot; a late answer just updates the target mid-flight.
- **Prompt format = study's sweet spot:** game state as `key=value` lines,
  output constrained to one JSON object. Study §6: semi-structured → JSON
  is "nailed it" territory. Few-shot examples counter the documented label
  bias (study's own recommendation).
- **Fallback ladder:** LLM → deterministic AI → center. Gameplay never
  stalls; model outage degrades to v1.

## 4. Technology choices

| Choice | Pick | Why |
|---|---|---|
| Model | **gemma3:270m** primary; **qwen2.5:0.5b** alternate; llama3.2:1b if quality insufficient | 270m is studied (~0.2–0.5s), qwen2.5:0.5b reportedly follows instructions better at same speed class |
| API | `POST /api/generate`, `stream:false`, `temperature 0.1`, `num_predict ≈ 48` | Parity with the benchmark; tiny output = fast |
| HTTP client | stdlib `urllib.request` in a thread | Keeps the game zero-dependency (`requests` stays a test tool) |
| "Cloud" | point `OLLAMA_URL` at the remote box; `keep_alive` prevents cold reloads | Ollama is self-hosted; the REST API is identical local or remote — adapter is one constant |

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| LLM can't do intercept math | Hybrid design; optional "pure mode" (LLM predicts intercept from raw state or ASCII board render — benchmark first) |
| Latency spike / outage | 1.0s timeout, seamless fallback; circuit breaker after 3 consecutive failures |
| Malformed JSON / out-of-range | Brace-extraction parser (study §6 pattern), clamp to field, validate before use |
| Label bias (all "aggressive") | Few-shot examples spanning the choice space, temperature 0.1 |
| Too weak / too strong opponent | Blend factor between fallback target and LLM target = difficulty knob |
| Thread safety | Single result attribute written by worker, read by loop; GIL-safe |

## 6. Operational concerns

- Startup health check (`GET /api/tags`); banner shows model + measured
  latency; if down → "fallback mode" notice.
- Debug flag logs `{state, prompt, response, latency}` to `.jsonl` for
  prompt tuning.
- Config constants at top of file, matching v1's tuning-table ethos.

## 7. Model recommendations

Measured against the live server (remote Ollama, ~470 ms ping):

| Model | Warm latency | Verdict |
|---|---|---|
| **gemma3:270m** | ~230 ms | Default. Answers within every rally; few-shot JSON output is reliable (study §6 sweet spot) |
| qwen3:0.6b / gemma3:1b | ~250–400 ms | Best quality-per-latency if 270m's offsets feel repetitive; still lands every rally |
| qwen2.5:3b … 7b | ~0.5–1.5 s | Borderline: answers arrive mid-flight; paddle commits late but still uses the LLM strike |
| cloud models (minimax-m3, kimi-k2, …) | 1–5 s+ | "Sometimes" opponent — most rallies fall back to deterministic centering; raise `--timeout` to 8+ |

Rules of thumb:

- **Latency budget:** ball flight after your hit is 1.0–3.2 s. A model must
  answer in ≲0.5 s to influence *every* rally; ≤1.5 s to influence most.
- **Output size dominates latency** — `num_predict: 48` keeps answers tiny;
  never raise it for big models, lower it.
- **Cold start ≈ 1.1 s** on first call — paid up front by `warm_model()`
  at launch; `keep_alive: 5m` holds the model resident between rallies.
- **Prompt format matters more than model size:** key=value state + few-shot
  JSON examples (the study's proven pattern) get more out of a 270m model
  than a bigger model with a vague prompt.
- **Label bias** (study §5) is the failure mode to watch: if the model
  always returns the same offset, add contrasting few-shot examples.

## 8. Tuning knobs (v2)

| Constant | Default | Meaning |
|---|---|---|
| `LLM_TIMEOUT` / `--timeout` | 1.5 s | Per-call budget; raise for slow cloud models |
| `LLM_BLEND` | 1.0 | 0 = pure deterministic centering, 1 = full LLM offset. Mid-values soften a wild model |
| `LLM_MAX_ADJUST` | 2.0 | Cells the LLM may shift its target; raise for a more mobile opponent |
| `LLM_BREAKER` | 3 | Consecutive failures before fallback-only mode |
| `LLM_NUM_PREDICT` | 48 | Max answer tokens; keep tiny for speed |
| `AI_MAX_SPEED` / `AI_ACCEL` | 20 / 90 | Paddle travel limits — the real difficulty ceiling; a slow paddle can't reach any intercept, LLM or not |
| `PONG_DEBUG=1` | off | Logs every prompt/response/latency to `pong_llm_debug.jsonl` for tuning |

Difficulty recipe: to make the LLM opponent *feel* stronger without cheating,
raise `AI_MAX_SPEED`/`AI_ACCEL` first (it can now reach the intercepts it
chooses), then `LLM_MAX_ADJUST`, then try a bigger model. To weaken it,
lower `AI_MAX_SPEED` — the LLM's choices stay visible but it misses more.

## 9. Cross-platform sound

macOS system sounds don't exist on Android/Termux or bare Linux, so v2
resolves audio at runtime:

- **Player detection** (first found wins): `afplay` →
  `termux-media-player` → sox `play` → `mpv` → `paplay` → `ffplay`.
- **macOS**: uses the original `/System/Library/Sounds/*.aiff` files.
- **Elsewhere**: synthesizes small WAV tones with the stdlib `wave`
  module (paddle 880 Hz, wall 440 Hz, serve 660 Hz, score = descending
  3-note jingle), cached in the temp dir, ~16 KB each, fade edges to
  avoid clicks. Still zero dependencies.
- **No player at all**: falls back to the terminal bell, silently.
- Android needs `pkg install termux-api` for `termux-media-player`.

## 10. Small-terminal support

`fit_field()` shrinks `WIDTH`/`HEIGHT` to the real terminal (floored at
20×8) before paddles/ball are created; all physics reads the globals
live, so nothing else changes. `KEY_RESIZE` re-fits mid-game. The probe
and model-picker screens compute their layout from `getmaxyx()`.

## 11. Next step

Benchmark the model's offset choices against the deterministic optimum
(reusing the `benchmark_gemma3_270m.py` pattern), and playtest to tune
`LLM_BLEND` and the few-shot examples from the debug log.
