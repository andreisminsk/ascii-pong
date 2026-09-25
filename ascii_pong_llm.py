#!/usr/bin/env python3
"""ASCII Pong v2 — the right paddle is driven by an Ollama LLM.

Architecture (see PONG-AI-ARCH.md): hybrid intelligence.
- Physics predicts WHERE the ball will cross the right paddle plane
  (deterministic forward simulation — exact, microseconds).
- The LLM decides HOW to strike: a hit offset in [-1, 1] (becomes spin
  via Ball.bounce) plus a small position adjustment. That is a judgment
  call in the small-model sweet spot: few-shot, key=value state in,
  one tiny JSON object out.
- Calls run on a daemon thread with a hard timeout; the game loop never
  blocks. Silence, malformed output, or outages degrade to the v1
  deterministic AI (center on the intercept); a circuit breaker
  disables the LLM after repeated failures.

Run:  python3 ascii_pong_llm.py [--model MODEL] [--timeout SECS]
Env:  OLLAMA_URL   (default http://localhost:11434)
      PONG_MODEL   (default gemma3:270m; --model overrides)
      PONG_DEBUG=1 -> log prompts/responses to pong_llm_debug.jsonl

Before each game a pre-flight probe checks the model (server reachable,
model listed, answers parseable JSON within the timeout) and asks how
to proceed if something is wrong: play anyway, CPU fallback, or pick
another model from the server.
"""
import argparse
import curses
import difflib
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave


# macOS system sounds (used when present); other platforms get
# synthesized WAV tones (see _TONES) played by whatever tool exists.
_SOUNDS = {
    'paddle': '/System/Library/Sounds/Tink.aiff',
    'wall':   '/System/Library/Sounds/Pop.aiff',
    'serve':  '/System/Library/Sounds/Tink.aiff',
    'score':  '/System/Library/Sounds/Submarine.aiff',
}

# synthesized tones: kind -> [(Hz, ms), ...]
_TONES = {
    'paddle': [(880, 60)],
    'wall':   [(440, 50)],
    'serve':  [(660, 80)],
    'score':  [(600, 120), (400, 100), (250, 150)],
}

_sound_paths = {}   # kind -> playable file (resolved on first beep)
_player = None      # command builder, or None -> terminal bell
_sounds_ready = False


def _detect_player():
    """First available audio player. afplay (macOS), Termux:API,
    sox, mpv, paplay, ffplay. Returns a path->argv builder or None."""
    candidates = (
        ('afplay', lambda p: ['afplay', p]),
        ('termux-media-player',
         lambda p: ['termux-media-player', 'play', p]),
        ('play', lambda p: ['play', '-q', p]),
        ('mpv', lambda p: ['mpv', '--really-quiet', '--no-video', p]),
        ('paplay', lambda p: ['paplay', p]),
        ('ffplay', lambda p: ['ffplay', '-nodisp', '-autoexit',
                               '-loglevel', 'quiet', p]),
    )
    for cmd, mk in candidates:
        if shutil.which(cmd):
            return mk
    return None


def _synth_wav(path, tones):
    """Write a small mono WAV for the given (Hz, ms) segments.
    Pure stdlib; ~4 ms fade edges avoid clicks."""
    rate = 22050
    samples = []
    fade = max(1, int(rate * 0.004))
    for freq, ms in tones:
        n = max(1, int(rate * ms / 1000))
        for i in range(n):
            env = min(1.0, i / fade, (n - i) / fade)
            samples.append(int(16000 * env
                               * math.sin(2 * math.pi * freq * i / rate)))
    with wave.open(path, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack('<%dh' % len(samples), *samples))


def _init_sounds():
    """Resolve a playable file per sound kind. macOS system sounds when
    present; synthesized WAVs elsewhere (Android/Termux, Linux)."""
    global _player, _sounds_ready
    _player = _detect_player()
    cache = os.path.join(tempfile.gettempdir(), 'ascii-pong-sounds')
    for kind, tones in _TONES.items():
        sys_path = _SOUNDS.get(kind)
        if sys_path and os.path.exists(sys_path):
            _sound_paths[kind] = sys_path
            continue
        if _player is None:
            continue
        try:
            os.makedirs(cache, exist_ok=True)
            path = os.path.join(cache, kind + '.wav')
            if not os.path.exists(path):
                _synth_wav(path, tones)
            _sound_paths[kind] = path
        except OSError:
            pass
    _sounds_ready = True


def beep(kind='paddle'):
    try:
        if not _sounds_ready:
            _init_sounds()
        path = _sound_paths.get(kind)
        if path and _player:
            subprocess.Popen(_player(path), stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
        else:
            curses.beep()
    except Exception:
        pass


def hide_cursor(stdscr):
    """Hide the cursor; some terminals can't and return ERR."""
    try:
        curses.curs_set(0)
    except curses.error:
        pass


WIDTH, HEIGHT = 60, 18
PADDLE_H = 4

# Paddle physics (cells, seconds)
IMPULSE = 14.0        # velocity gained per key press
MAX_SPEED = 26.0      # player paddle speed cap
FRICTION = 4.0        # drag: higher = stops sooner
AI_MAX_SPEED = 20.0   # AI is a bit slower than you
AI_ACCEL = 90.0       # cells/s^2 the AI pushes with

# Ball physics
BALL_SPEED0 = 18.0
BALL_SPEEDUP = 2.0
BALL_SPEEDMAX = 55.0
SPIN = 0.8            # max fraction of speed turned into vertical motion

# LLM opponent (v2)
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
LLM_MODEL = os.environ.get('PONG_MODEL', 'gemma3:270m')
LLM_TIMEOUT = 1.5     # s; warm calls ~0.25s, cold ~1.1s; flight is 1.0-3.2s
LLM_BLEND = 1.0       # 0 = pure deterministic centering, 1 = full LLM offset
LLM_MAX_ADJUST = 2.0  # cells the LLM may shift its paddle target
LLM_BREAKER = 3       # consecutive failures before fallback-only mode
LLM_REARM = 20.0      # s of fallback before the breaker retries the LLM
LLM_NUM_PREDICT = 48  # tiny answers = fast
DEBUG_LOG = 'pong_llm_debug.jsonl'


def dbg(**rec):
    """Append a record to the debug log when PONG_DEBUG is set."""
    if not os.environ.get('PONG_DEBUG'):
        return
    rec['ts'] = round(time.time(), 3)
    try:
        with open(DEBUG_LOG, 'a') as f:
            f.write(json.dumps(rec) + '\n')
    except OSError:
        pass


class Paddle:
    def __init__(self, y, vcap=MAX_SPEED):
        self.y = float(y)
        self.vy = 0.0
        self.vcap = vcap

    def push(self, dv):
        self.vy = max(-self.vcap, min(self.vcap, self.vy + dv))

    def step(self, dt):
        self.vy *= math.exp(-FRICTION * dt)
        self.y += self.vy * dt
        top = HEIGHT - PADDLE_H
        if self.y < 0:
            self.y, self.vy = 0.0, 0.0
        elif self.y > top:
            self.y, self.vy = float(top), 0.0

    def hits(self, by):
        # padded to match the drawn cells (ball floats between grid rows)
        return self.y - 0.75 <= by <= self.y + PADDLE_H - 0.25


class Ball:
    def __init__(self, serve_left=None):
        self.reset(serve_left)

    def reset(self, serve_left=None, paddle_y=None):
        """Serve from the middle of the server's paddle, aimed into the field.
        serve_left=True: left border; False: right border; None: random."""
        self.waiting = True   # ball held until serve
        if serve_left is None:
            serve_left = random.choice((True, False))
        self.x = 1.0 if serve_left else float(WIDTH - 2)
        # ball rests at the middle of the serving paddle
        self.y = (paddle_y + PADDLE_H / 2) if paddle_y is not None \
            else HEIGHT / 2 - 0.5
        # randomized serve direction (angle up to ~60 degrees)
        ang = random.uniform(-1.0, 1.0)
        d = 1 if serve_left else -1
        self.vx = math.cos(ang) * BALL_SPEED0 * d
        self.vy = math.sin(ang) * BALL_SPEED0

    def serve(self):
        """Launch a waiting ball."""
        if self.waiting:
            self.waiting = False

    def bounce(self, paddle, dirx):
        rel = (self.y - (paddle.y + PADDLE_H / 2)) / (PADDLE_H / 2)
        rel = max(-1.0, min(1.0, rel))
        speed = min(math.hypot(self.vx, self.vy) + BALL_SPEEDUP, BALL_SPEEDMAX)
        self.vy = rel * speed * SPIN
        self.vx = dirx * math.sqrt(max(speed * speed - self.vy * self.vy, 1.0))

    def step(self, dt, p1, p2):
        """Returns 'paddle' | 'wall' | 'score' | None.
        Uses swept collision so a fast ball can't tunnel past a paddle."""
        old_x, old_y = self.x, self.y
        self.x += self.vx * dt
        self.y += self.vy * dt
        hit = None
        if self.y <= 0:
            self.y, self.vy = 0.0, abs(self.vy)
            hit = 'wall'
        elif self.y >= HEIGHT - 1:
            self.y, self.vy = float(HEIGHT - 1), -abs(self.vy)
            hit = 'wall'
        # Interpolate ball's y when it crossed each paddle plane
        if self.vx < 0 and old_x > 1 >= self.x:
            f = (old_x - 1) / (old_x - self.x) if old_x != self.x else 0
            y_at = old_y + (self.y - old_y) * f
            if p1.hits(y_at):
                self.y = y_at
                self.bounce(p1, 1)
                self.x = 1.0
                hit = 'paddle'
        elif self.vx > 0 and old_x < WIDTH - 2 <= self.x:
            plane = WIDTH - 2
            f = (plane - old_x) / (self.x - old_x) if self.x != old_x else 0
            y_at = old_y + (self.y - old_y) * f
            if p2.hits(y_at):
                self.y = y_at
                self.bounce(p2, -1)
                self.x = float(plane)
                hit = 'paddle'
        if self.x < 0 or self.x > WIDTH - 1:
            return 'score'
        return hit


# ── deterministic half of the hybrid: WHERE to intercept ────────

def predict_intercept(ball):
    """Forward-simulate a copy of the ball to the right paddle plane,
    mirroring Ball.step's wall reflections exactly.
    Returns (y_at_plane, seconds_until, vy_at_plane) or (None, None, None)."""
    if ball.vx <= 0:
        return None, None, None
    x, y, vx, vy = ball.x, ball.y, ball.vx, ball.vy
    plane = float(WIDTH - 2)
    dt = 0.005
    t = 0.0
    while t < 6.0:
        nx = x + vx * dt
        ny = y + vy * dt
        if ny <= 0.0:
            ny, vy = 0.0, abs(vy)
        elif ny >= HEIGHT - 1:
            ny, vy = float(HEIGHT - 1), -abs(vy)
        if nx >= plane:
            f = (plane - x) / (nx - x)
            return y + (ny - y) * f, t + dt * f, vy
        x, y = nx, ny
        t += dt
    return None, None, None


# ── LLM half of the hybrid: HOW to strike ───────────────────────

SYSTEM = (
    "You are the right paddle in pong. The ball is flying toward you. "
    "Choose your strike. Answer with ONLY one JSON object, nothing else: "
    '{"offset": <number>, "adjust": <number>}\n'
    "offset: -1..1. Negative = ball strikes your upper half, returns UP. "
    "Positive = lower half, returns DOWN. 0 = center, straight and fast.\n"
    "adjust: -2..2, a small paddle position shift in cells.\n"
    "Tactic: return the ball AWAY from where the human paddle is now."
)

FEWSHOT = """Example 1:
ball_y=4.0 ball_vy=-6.0 human_y=3.0 paddle_y=6.0 time_left=1.5 score=3:2 rally=4
{"offset": 0.8, "adjust": 0.0}

Example 2:
ball_y=14.0 ball_vy=5.0 human_y=13.0 paddle_y=8.0 time_left=1.2 score=3:3 rally=2
{"offset": -0.8, "adjust": 0.0}

Example 3:
ball_y=9.0 ball_vy=0.0 human_y=9.0 paddle_y=8.0 time_left=0.4 score=10:10 rally=6
{"offset": 0.3, "adjust": 0.5}

Example 4:
ball_y=16.0 ball_vy=-8.0 human_y=15.0 paddle_y=14.0 time_left=0.9 score=0:1 rally=1
{"offset": -0.6, "adjust": -0.5}

Now:"""


def build_prompt(ctrl, p1, p2, score, rally):
    """key=value state — the format small models handle best."""
    return (FEWSHOT + "\n"
            f"ball_y={ctrl.intercept_y:.1f} ball_vy={ctrl.vy_at:.1f} "
            f"human_y={p1.y + PADDLE_H / 2:.1f} "
            f"paddle_y={p2.y + PADDLE_H / 2:.1f} "
            f"time_left={ctrl.time_left:.1f} "
            f"score={score[1]}:{score[0]} rally={rally}")


def parse_llm(text):
    """Extract {offset, adjust} from a response; clamp to sane ranges.
    Returns (offset, adjust) or None if unusable."""
    s, e = text.find('{'), text.rfind('}')
    if s < 0 or e <= s:
        return None
    try:
        obj = json.loads(text[s:e + 1])
        off = float(obj.get('offset', 0.0))
        adj = float(obj.get('adjust', 0.0))
    except (ValueError, TypeError, AttributeError):
        return None
    off = max(-1.0, min(1.0, off))
    adj = max(-LLM_MAX_ADJUST, min(LLM_MAX_ADJUST, adj))
    return off, adj


class AsyncAdvisor:
    """One-shot LLM calls from daemon threads; the game loop never blocks.
    Latest result wins; a circuit breaker falls back after LLM_BREAKER
    consecutive failures."""

    def __init__(self):
        self.url = OLLAMA_URL.rstrip('/') + '/api/generate'
        self.enabled = True
        self.user_disabled = False   # 'f' menu choice - stays off
        self.tripped_at = None       # when the breaker last tripped
        self.failures = 0
        self.seq = 0
        self.result = None        # (seq, offset, adjust, latency_ms)
        self._delivered = 0
        self._lock = threading.Lock()

    def submit(self, prompt):
        """Fire a request; returns its sequence number, or None if disabled."""
        with self._lock:
            if not self.enabled:
                return None
            self.seq += 1
            seq = self.seq
        threading.Thread(target=self._worker, args=(seq, prompt),
                         daemon=True).start()
        return seq

    def _worker(self, seq, prompt):
        payload = json.dumps({
            'model': LLM_MODEL,
            'prompt': prompt,
            'system': SYSTEM,
            'stream': False,
            'keep_alive': '5m',
            'options': {'temperature': 0.1, 'num_predict': LLM_NUM_PREDICT},
        }).encode()
        t0 = time.time()
        try:
            req = urllib.request.Request(
                self.url, data=payload,
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
                out = json.loads(r.read()).get('response', '')
            parsed = parse_llm(out)
            if parsed is None:
                raise ValueError('unparseable: %r' % out[:80])
            ms = (time.time() - t0) * 1000
            with self._lock:
                self.failures = 0
                self.result = (seq, parsed[0], parsed[1], ms)
            dbg(seq=seq, ms=round(ms), ok=True, prompt=prompt, response=out)
        except Exception as e:
            with self._lock:
                self.failures += 1
                if self.failures >= LLM_BREAKER and self.enabled:
                    self.enabled = False
                    self.tripped_at = time.time()
            dbg(seq=seq, ok=False, error=str(e))

    def collect(self):
        """Return the newest undelivered result, if any."""
        with self._lock:
            if self.result and self.result[0] > self._delivered:
                self._delivered = self.result[0]
                return self.result
        return None

    def rearm(self):
        """Fresh chance for the LLM (new game). Honors user_disabled."""
        with self._lock:
            self.failures = 0
            self.tripped_at = None
            if not self.user_disabled:
                self.enabled = True

    def maybe_rearm(self):
        """Mid-game: retry the LLM after LLM_REARM seconds of fallback."""
        with self._lock:
            if (not self.enabled and not self.user_disabled
                    and self.tripped_at is not None
                    and time.time() - self.tripped_at >= LLM_REARM):
                self.enabled = True
                self.failures = 0
                self.tripped_at = None


ADVISOR = AsyncAdvisor()


class AIController:
    """Right-paddle brain: deterministic intercept target + LLM strike choice.
    While the LLM is thinking (or absent), the paddle centers on the
    predicted intercept — exactly the v1 AI, so play never stalls."""

    def __init__(self):
        self.intercept_y = None   # predicted crossing at the right plane
        self.vy_at = 0.0
        self.time_left = 0.0
        self.offset = 0.0         # LLM strike offset (-1..1)
        self.adjust = 0.0         # LLM target shift (cells)
        self.hit_seq = -1         # sequence number of the pending request
        self.last_ms = None      # latency of the last applied answer

    def on_player_hit(self, ball):
        y, t, vy = predict_intercept(ball)
        self.intercept_y, self.time_left, self.vy_at = y, t, vy
        self.offset = self.adjust = 0.0

    def on_llm(self, seq, offset, adjust, ms):
        if seq == self.hit_seq:   # drop answers to superseded requests
            self.offset, self.adjust, self.last_ms = offset, adjust, ms

    def new_point(self):
        self.intercept_y = None
        self.offset = self.adjust = 0.0

    def target(self, ball_vx, waiting):
        """Paddle-top position to steer toward."""
        if waiting or ball_vx <= 0 or self.intercept_y is None:
            return HEIGHT / 2 - PADDLE_H / 2   # home
        # place the paddle so the ball meets it at the LLM's offset:
        # rel = (ball_y - paddle_center) / (PADDLE_H/2)  ->  Ball.bounce
        t = (self.intercept_y - PADDLE_H / 2
             - LLM_BLEND * self.offset * PADDLE_H / 2 + self.adjust)
        return max(0.0, min(float(HEIGHT - PADDLE_H), t))


def sample_prompt():
    """A representative game prompt, used by the pre-flight probe."""
    c = AIController()
    c.intercept_y, c.vy_at, c.time_left = 7.3, -4.0, 1.4
    return build_prompt(c, Paddle(3.0), Paddle(6.0), [2, 3], 4)


def probe_model():
    """Pre-flight diagnostic of LLM_MODEL. Returns a report dict:
      reachable, listed, names, ping_ms,
      calls: [{ms, ok, done, raw}] - two live calls with the game prompt,
      problems: human-readable issues (empty = model is game-ready),
      ok: True when problems is empty."""
    rep = {'reachable': False, 'listed': False, 'names': [],
           'ping_ms': None, 'calls': [], 'problems': []}
    try:
        t0 = time.time()
        with urllib.request.urlopen(OLLAMA_URL.rstrip('/') + '/api/tags',
                                    timeout=3) as r:
            rep['names'] = [m.get('name', '')
                           for m in json.loads(r.read()).get('models', [])]
        rep['ping_ms'] = (time.time() - t0) * 1000
        rep['reachable'] = True
    except Exception:
        rep['problems'].append('server unreachable at ' + OLLAMA_URL)
        return rep
    rep['listed'] = any(n == LLM_MODEL or n.split(':')[0] ==
                        LLM_MODEL.split(':')[0] for n in rep['names'])
    if not rep['listed']:
        near = difflib.get_close_matches(LLM_MODEL, rep['names'],
                                         n=3, cutoff=0.3)
        msg = 'model not on server'
        if near:
            msg += ' - did you mean: ' + ', '.join(near)
        rep['problems'].append(msg)
        return rep
    # Two live calls with the real game prompt (the first may be cold).
    url = OLLAMA_URL.rstrip('/') + '/api/generate'
    for _ in range(2):
        call = {'ms': None, 'ok': False, 'done': '', 'raw': ''}
        try:
            payload = json.dumps({
                'model': LLM_MODEL, 'prompt': sample_prompt(),
                'system': SYSTEM, 'stream': False, 'keep_alive': '5m',
                'options': {'temperature': 0.1,
                            'num_predict': LLM_NUM_PREDICT},
            }).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={'Content-Type': 'application/json'})
            t0 = time.time()
            with urllib.request.urlopen(
                    req, timeout=max(LLM_TIMEOUT * 4, 10)) as r:
                body = json.loads(r.read())
            call['ms'] = (time.time() - t0) * 1000
            call['raw'] = body.get('response', '')
            call['done'] = str(body.get('done_reason', ''))
            call['ok'] = parse_llm(call['raw']) is not None
        except Exception as e:
            call['done'] = type(e).__name__
        rep['calls'].append(call)
    good = [c for c in rep['calls'] if c['ok']]
    if not good:
        if any(c['done'] == 'length' and not c['raw']
               for c in rep['calls']):
            rep['problems'].append(
                'reasoning model: thinks past its token budget and never '
                'answers (done_reason=length, empty response) - pick a '
                'non-reasoning model')
        elif any(c['done'] in ('TimeoutError', 'HTTPError', 'URLError')
                 for c in rep['calls']):
            rep['problems'].append(
                'calls failed (timeout/server error) - model or server '
                'too slow for the probe budget')
        else:
            last = rep['calls'][-1]
            rep['problems'].append(
                'output not parseable as {"offset", "adjust"}: '
                + repr((last['raw'] or last['done'])[:60]))
    else:
        ms = good[-1]['ms']
        if ms > LLM_TIMEOUT * 1000:
            rep['problems'].append(
                f'answers take ~{ms:.0f} ms, longer than --timeout '
                f'{LLM_TIMEOUT:.1f}s - most rallies will fall back to '
                'the CPU paddle; raise --timeout')
    rep['ok'] = not rep['problems']
    return rep


def pick_model(stdscr, names):
    """Interactive model picker (numbered, paginated, terminal-aware).
    Returns the chosen model name, or None if cancelled."""
    rows, cols = stdscr.getmaxyx()
    body = max(1, rows - 4)          # rows available for the list
    ncols = max(1, cols // 26)       # model names per row
    per_page = body * ncols
    page, sel = 0, ''
    pages = max(1, math.ceil(len(names) / per_page))
    while True:
        stdscr.erase()
        try:
            stdscr.addstr(0, 2, f"pick a model - {len(names)} on server, "
                                f"page {page + 1}/{pages}  "
                                "(number+Enter, n/p pages, Esc cancels)"
                                [:cols - 1])
        except curses.error:
            pass
        for i, n in enumerate(names[page * per_page:
                                   (page + 1) * per_page]):
            num = page * per_page + i + 1
            try:
                stdscr.addstr(2 + i % body, 2 + (i // body) * 26,
                              f"{num:>2} {n[:21]}")
            except curses.error:
                pass
        try:
            stdscr.addstr(rows - 2, 2, f"> {sel}")
        except curses.error:
            pass
        stdscr.refresh()
        k = stdscr.getch()
        if ord('0') <= k <= ord('9') and len(sel) < 4:
            sel += chr(k)
        elif k in (8, 127, curses.KEY_BACKSPACE):
            sel = sel[:-1]
        elif k in (10, 13, curses.KEY_ENTER) and sel:
            i = int(sel) - 1
            if 0 <= i < len(names):
                return names[i]
            sel = ''
        elif k in (ord('n'), ord('N'), curses.KEY_NPAGE):
            page = (page + 1) % pages
        elif k in (ord('p'), ord('P'), curses.KEY_PPAGE):
            page = (page - 1) % pages
        elif k in (27, ord('q'), ord('Q')):
            return None


def confirm_probe(stdscr, rep):
    """Show the probe report; get the user's decision.
    Returns 'play' | 'fallback' | 'model:<name>' | 'quit'.
    Auto-continues when the model is game-ready."""
    stdscr.nodelay(False)          # blocking input for the menus
    stdscr.timeout(-1)

    def draw():
        stdscr.erase()
        y = 1

        def line(s='', attr=0):
            nonlocal y
            try:
                stdscr.addstr(y, 2, s, attr)
            except curses.error:
                pass
            y += 1

        line(f"model probe: {LLM_MODEL}  ({OLLAMA_URL})")
        if rep['reachable']:
            line(f"server: reachable, ping {rep['ping_ms']:.0f} ms, "
                 f"{len(rep['names'])} models")
        for c in rep['calls']:
            mark = 'ok ' if c['ok'] else 'BAD'
            ms = f"{c['ms']:.0f} ms" if c['ms'] is not None else 'failed'
            line(f"test call: {mark}  {ms}  done={c['done'] or '?'}")
        if rep['ok']:
            line()
            line("model ready - starting in 3 s (any key to start now) ...")
        else:
            line()
            line("problems found:", curses.A_BOLD)
            for p in rep['problems']:
                line("  - " + p)
            line()
            line("[Enter] play anyway   [f] CPU fallback   "
                 + ("[m] pick model   " if rep['names'] else "")
                 + "[q] quit")
        stdscr.refresh()

    draw()
    if rep['ok']:
        # guaranteed readable pause before the field appears;
        # any keypress starts immediately
        stdscr.timeout(3000)          # ms
        stdscr.getch()                # waits up to 3 s
        stdscr.timeout(-1)
        return 'play'
    while True:
        k = stdscr.getch()
        if k in (10, 13, curses.KEY_ENTER):
            return 'play'
        if k in (ord('f'), ord('F')):
            return 'fallback'
        if k in (ord('m'), ord('M')) and rep['names']:
            name = pick_model(stdscr, rep['names'])
            if name:
                return 'model:' + name
            draw()                 # cancelled: back to the report
        if k in (ord('q'), ord('Q'), 27):
            return 'quit'


def fit_field(stdscr):
    """Shrink the field to fit the actual terminal (Termux/phones are
    narrow). Must run before the game's paddles/ball are created."""
    global WIDTH, HEIGHT
    rows, cols = stdscr.getmaxyx()
    WIDTH = max(20, min(WIDTH, cols - 2))
    HEIGHT = max(8, min(HEIGHT, rows - 3))


def play_game(stdscr):
    """One game to 11. Returns True for rematch, False for quit."""
    fit_field(stdscr)      # adapt to the real terminal size
    stdscr.nodelay(True)   # restore non-blocking input after any game-end prompt
    stdscr.timeout(80)
    p1 = Paddle((HEIGHT - PADDLE_H) / 2)
    p2 = Paddle((HEIGHT - PADDLE_H) / 2, AI_MAX_SPEED)
    server = random.choice((0, 1))            # 0 = you (left), 1 = AI (right)
    ball = Ball()
    ball.reset(serve_left=(server == 0), paddle_y=(p1.y if server == 0 else p2.y))
    score = [0, 0]
    rally = 0
    ctrl = AIController()
    ADVISOR.rearm()               # new game: give the LLM a fresh chance
    ai_serve_at = time.time() + 1.2           # AI serves ~1.2s after reset
    last = time.time()

    while True:
        now = time.time()
        dt = min(now - last, 0.2)
        last = now

        # Input: drain all pending keys, fully non-blocking
        stdscr.nodelay(True)   # never wait inside the drain loop
        while True:
            k = stdscr.getch()
            if k == -1:
                break
            if k == curses.KEY_RESIZE:
                fit_field(stdscr)   # terminal resized mid-game: re-fit
                continue
            if k in (ord('q'), 27):
                return False
            if k in (ord('w'), ord('W'), ord('g'), ord('G'),
                     curses.KEY_UP):
                p1.push(-IMPULSE)
            elif k in (ord('s'), ord('S'), ord('v'), ord('V'),
                     curses.KEY_DOWN):
                p1.push(IMPULSE)
            elif k in (ord(' '), 10) and ball.waiting and server == 0:
                ball.serve()
                # a serve IS the player's hit: no 'paddle' event will fire,
                # so consult the LLM right now
                rally += 1
                ctrl.on_player_hit(ball)
                seq = ADVISOR.submit(
                    build_prompt(ctrl, p1, p2, score, rally))
                ctrl.hit_seq = seq if seq is not None else -1

        # Right paddle: steer toward the controller's target
        ADVISOR.maybe_rearm()      # breaker cool-off: retry the LLM
        res = ADVISOR.collect()
        if res:
            ctrl.on_llm(*res)
        want = ctrl.target(ball.vx, ball.waiting) + PADDLE_H / 2
        diff = want - (p2.y + PADDLE_H / 2)
        if abs(diff) > 0.8:
            p2.push(AI_ACCEL * dt * (1 if diff > 0 else -1))

        # Paddle physics: integrate velocity into position
        p1.step(dt)
        p2.step(dt)

        # Ball rides the serving paddle while waiting
        if ball.waiting:
            pad = p1 if server == 0 else p2
            ball.x = 1.0 if server == 0 else float(WIDTH - 2)
            ball.y = pad.y + PADDLE_H / 2

        # AI serves automatically after its delay
        if ball.waiting and server == 1 and time.time() >= ai_serve_at:
            ball.serve()
            beep('serve')

        if ball.waiting:
            scorer = None
        else:
            event = ball.step(dt, p1, p2)
            if event == 'paddle':
                beep('paddle')
                if ball.vx > 0:            # heading to the AI — consult the LLM
                    rally += 1
                    ctrl.on_player_hit(ball)
                    seq = ADVISOR.submit(
                        build_prompt(ctrl, p1, p2, score, rally))
                    ctrl.hit_seq = seq if seq is not None else -1
            elif event == 'wall':
                beep('wall')
            scorer = 0 if event == 'score' and ball.vx > 0 else \
                     1 if event == 'score' else None
        if scorer is not None:
            score[scorer] += 1
            # Game to 11, win by 2 (official rules)
            beep('score')
            if score[scorer] >= 11 and score[scorer] - score[1 - scorer] >= 2:
                stdscr.nodelay(False)
                stdscr.erase()
                msg = "You win!" if scorer == 0 else "AI wins!"
                x1 = max(0, WIDTH // 2 - 8)
                x2 = max(0, WIDTH // 2 - 22)
                stdscr.addstr(HEIGHT // 2, x1, msg)
                stdscr.addstr(HEIGHT // 2 + 1, x2,
                              f"Final {score[0]}:{score[1]}   "
                              f"R = rematch, Q = quit"[:max(0, WIDTH - 1 - x2)])
                while True:
                    k = stdscr.getch()
                    if k in (ord('r'), ord('R')):
                        return True
                    if k in (ord('q'), ord('Q'), 27):
                        return False
            # Serve rotation: every 2 points; every point at 10-10+
            if score[0] >= 10 and score[1] >= 10:
                server = 1 - server
            elif (score[0] + score[1]) % 2 == 0:
                server = 1 - server
            # serve from the middle of the new server's paddle
            ball.reset(serve_left=(server == 0),
                       paddle_y=(p1.y if server == 0 else p2.y))
            rally = 0
            ctrl.new_point()
            ai_serve_at = time.time() + 1.2
            time.sleep(0.4)
            last = time.time()

        # Draw
        stdscr.erase()
        srv = "You" if server == 0 else "AI"
        hint = "  SPACE to serve" if ball.waiting and server == 0 else ""
        if ADVISOR.enabled:
            mode = f"llm {ctrl.last_ms:.0f}ms" if ctrl.last_ms else "llm"
        else:
            mode = "cpu"
        stdscr.addstr(0, 0, (f" You {score[0]}:{score[1]} AI[{mode}]  "
                             f"serve: {srv}  (W/S/G/V, q=quit){hint}"
                             )[:WIDTH - 1])
        for row in range(1, HEIGHT + 1):
            stdscr.addch(row, 0, '|')
            stdscr.addch(row, WIDTH - 1, '|')
        for dy in range(PADDLE_H):
            stdscr.addch(int(p1.y) + dy + 1, 1, '#')
            stdscr.addch(int(p2.y) + dy + 1, WIDTH - 2, '#')
        by = max(1, min(HEIGHT, int(round(ball.y)) + 1))
        bx = max(1, min(WIDTH - 2, int(round(ball.x))))
        stdscr.addch(by, bx, 'O')
        # status line under the field: which model is playing
        tag = f" model: {LLM_MODEL}" + ("" if ADVISOR.enabled else " (fallback)")
        stdscr.addstr(HEIGHT + 1, 0, tag[:WIDTH - 1])
        stdscr.refresh()


def main(stdscr):
    global LLM_MODEL
    hide_cursor(stdscr)
    stdscr.keypad(True)
    stdscr.nodelay(True)
    stdscr.timeout(80)
    fit_field(stdscr)               # adapt before any screen is drawn
    while True:
        rows, cols = stdscr.getmaxyx()
        stdscr.erase()
        try:
            stdscr.addstr(max(0, rows // 2 - 1), 2,
                          f"probing {LLM_MODEL} at {OLLAMA_URL} ..."
                          [:cols - 1])
            stdscr.addstr(max(0, rows // 2), 2,
                          "(two test calls - may take a while for slow "
                          "models)"[:cols - 1])
        except curses.error:
            pass
        stdscr.refresh()
        rep = probe_model()
        action = confirm_probe(stdscr, rep)
        if action == 'play':
            ADVISOR.enabled = True
            break
        if action == 'fallback':
            ADVISOR.enabled = False
            ADVISOR.user_disabled = True   # stays off, even across games
            break
        if action == 'quit':
            return
        if action.startswith('model:'):
            LLM_MODEL = action[6:]     # re-probe the new choice
    while play_game(stdscr):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="ASCII Pong vs an Ollama LLM paddle")
    ap.add_argument('--model', default=LLM_MODEL, metavar='MODEL',
                    help="Ollama model driving the right paddle "
                         "(default: %(default)s)")
    ap.add_argument('--timeout', type=float, default=LLM_TIMEOUT,
                    metavar='SECS',
                    help="per-call LLM timeout in seconds "
                         "(default: %(default)s)")
    args = ap.parse_args()
    LLM_MODEL = args.model          # rebind globals; workers read them live
    LLM_TIMEOUT = args.timeout
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
