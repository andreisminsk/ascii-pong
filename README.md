# ASCII Pong

Two single-file terminal ping-pong games written in pure Python
(standard library only — no dependencies).

## Games

| File | Opponent |
|---|---|
| `ascii_pong.py` | Deterministic CPU paddle (v1) |
| `ascii_pong_llm.py` | LLM-driven paddle via Ollama (v2) |

## Run

```sh
python3 ascii_pong.py            # classic, deterministic AI
python3 ascii_pong_llm.py        # LLM opponent
```

Any POSIX terminal (macOS Terminal, iTerm2, Linux) works.
Windows: use WSL.

## Controls

| Key          | Action                          |
|--------------|---------------------------------|
| `W` `G` `F` `H` `T` `U` / `↑` | Move paddle up (with momentum)  |
| `S` `V` `C` `B` / `↓`         | Move paddle down (with momentum)|
| `SPACE`      | Serve (when it's your serve); pause/resume during play |
| `Q` / `Esc`  | Quit                            |
| `R`          | Rematch (at game end)           |

While you hold the serve, the ball rides the middle of your paddle —
move the paddle to choose where to serve from.

## Game rules

Official table-tennis scoring:

- Game to **11 points**, win by **2**.
- First server chosen randomly; then serve alternates **every 2 points**.
- At **10–10 (deuce)**: serve alternates **every point**, game continues
  until someone leads by 2.
- The player who conceded the point does not serve — rotation follows
  the rules above regardless of who scored.

## Prize

Win a game (either version) and `aryna.txt` — ASCII art — is printed
line by line as a reward, cut to the terminal width (never wrapped).
The game temporarily leaves curses so the art lands in the terminal's
normal scrollback: it stays visible, the rematch prompt appears right
below it, and you can scroll back through everything. Pace:
`PRIZE_DELAY` (0.15 s/line).

## Physics

- Paddles have momentum: keys give velocity impulses, drag (friction)
  slows them, so they accelerate, glide, and coast to a stop.
- The ball is a continuous (float) simulation integrated per frame;
  rendering rounds it to terminal cells.
- **Spin**: bounce angle depends on where the ball strikes the paddle —
  hit off-center to angle shots.
- **Rally speed**: each paddle hit speeds the ball up (capped).
- **Swept collision**: fast balls can't tunnel through paddles —
  collision is checked along the whole frame's path.

## AI

The right paddle accelerates toward the ball with capped speed
(slightly slower than you), so it can be outplayed with fast
direction changes and spin.

## Sounds (macOS)

Paddle hit: *Tink* · wall bounce: *Pop* · point scored: *Submarine*,
played via `afplay` in a background process. Silently ignored
elsewhere; swap paths in `_SOUNDS` for other sounds
(`ls /System/Library/Sounds/`).

## Tuning

All knobs are constants at the top of `ascii_pong.py`:

| Constant        | Meaning                                   |
|----------------|-------------------------------------------|
| `WIDTH/HEIGHT`  | Field size (cells)                        |
| `PADDLE_H`      | Paddle length                              |
| `IMPULSE`       | Velocity gained per keypress              |
| `MAX_SPEED`     | Player paddle speed cap                   |
| `FRICTION`      | Drag — higher = stops sooner              |
| `AI_MAX_SPEED`  | AI speed cap (lower = easier)             |
| `AI_ACCEL`      | AI acceleration (lower = easier)          |
| `BALL_SPEED0`   | Serve speed                               |
| `BALL_SPEEDUP`  | Speed gained per paddle hit               |
| `BALL_SPEEDMAX` | Rally speed cap                           |
| `SPIN`          | How much of speed turns into vertical motion |
| `PONG_SPEED`    | Env var: global pace dial; default = WIDTH/60, so narrow terminals auto-slow |
| `+` / `-`       | In-game: speed up / slow down (×1.25 / ×0.8, clamped 0.4–2.5×) — main keyboard and numpad |

## LLM Opponent (v2)

`ascii_pong_llm.py` replaces the right paddle's brain with a model
served by [Ollama](https://ollama.com) — local or remote, same REST API.

### How it plays

Hybrid intelligence (see `PONG-AI-ARCH.md` for the full design):

- **Physics decides where** — a deterministic forward simulation predicts
  exactly where the ball will cross the paddle plane (including wall
  bounces), so the paddle always knows where to go.
- **The LLM decides how** — right after your hit, the model is asked to
  choose a strike: an offset in `[-1, 1]` (where the ball meets the
  paddle → spin) plus a small position adjustment. Few-shot prompt,
  `key=value` state in, one tiny JSON object out.
- **Never blocks** — calls run on a daemon thread with a hard timeout;
  while the model thinks, the paddle centers on the predicted intercept
  (v1 behavior), then shifts to the LLM's chosen strike when the answer
  lands. Late answers just retarget mid-flight.

### Run

```sh
python3 ascii_pong_llm.py                          # default: gemma3:270m
python3 ascii_pong_llm.py --model qwen3:0.6b       # any model on the server
python3 ascii_pong_llm.py --model glm-5.3:cloud --timeout 8
OLLAMA_URL=http://box:11434 python3 ascii_pong_llm.py
PONG_DEBUG=1 python3 ascii_pong_llm.py            # log prompts/responses
```

### Pre-flight probe

Before the first serve the game probes the model with two live test
calls and shows a report: server ping, per-call latency, parseability.
If something is wrong (model missing, reasoning model that never
answers, latency over `--timeout`) you choose:

| Key | Action |
|---|---|
| `Enter` | Play anyway |
| `f` | CPU fallback for the whole session |
| `m` | Pick another model from the server (paginated list) |
| `q` | Quit |

### Reliability

- 3 consecutive failures → circuit breaker → CPU paddle for 20 s,
  then the LLM gets a fresh chance (self-heals transient outages).
- Breaker fully reset on every new game.
- `f` (user choice) is the only permanent fallback.
- Header shows the live mode: `AI[llm 230ms]` or `AI[cpu]`.

### Model choice

Latency is the whole game — the ball gives the model 1.0–3.2 s to
answer, and it must reply in ≲0.5 s to influence every rally:

| Model | Warm latency | Verdict |
|---|---|---|
| `gemma3:270m` | ~230 ms | Default; answers every rally |
| `qwen3:0.6b`, `gemma3:1b` | ~250–400 ms | Slightly smarter, still every rally |
| 3b–7b models | 0.5–1.5 s | Commits late but still uses the LLM strike |
| Cloud models | 1–5 s+ | "Sometimes" opponent; raise `--timeout` |

Reasoning models (e.g. `glm-5.3:cloud`) burn their token budget on
thinking and never answer — the probe detects this and warns you.

### Tuning

| Constant | Default | Meaning |
|---|---|---|
| `LLM_TIMEOUT` / `--timeout` | 1.5 s | Per-call budget |
| `LLM_BLEND` | 1.0 | 0 = pure centering, 1 = full LLM offset |
| `LLM_MAX_ADJUST` | 2.0 | Cells the LLM may shift its target |
| `LLM_BREAKER` | 3 | Failures before fallback |
| `LLM_REARM` | 20 s | Fallback cool-off before retrying the LLM |
| `LLM_NUM_PREDICT` | 48 | Answer token cap — keep tiny |

## Small terminals & Android

Both games shrink the field to fit the real terminal (`fit_field`,
floored at 20×8) — works in Termux on phones. Resizing the window
mid-game re-fits the field. v2's probe and model-picker screens are
also terminal-aware.

### Sounds on Android/Linux

macOS system sounds don't exist there, so the game detects an audio
player at runtime (`termux-media-player`, sox `play`, `mpv`, `paplay`,
`ffplay`) and synthesizes its own WAV tones with the stdlib `wave`
module — still zero dependencies. On Android install:

```sh
pkg install termux-api
```

Without any player it falls back to the terminal bell, silently.

### Extra keys

`G` = up, `V` = down (aliases for W/S — handy on phone keyboards).
