#!/usr/bin/env python3
"""ASCII Pong with paddle physics.

Keys give paddles an impulse (velocity), which decays via friction —
so paddles accelerate, glide, and coast to a stop. Ball bounce angle
depends on where it hits the paddle, and rallies speed the ball up.
"""
import curses
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time
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


def flash():
    try:
        curses.flash()
    except curses.error:
        pass

PRIZE_DELAY = 0.15     # s per line when the win prize is revealed


def win_screen(score):
    """Human won: leave curses and print the prize art + result as
    regular terminal output, so it lands in the scrollback and stays
    put (the rematch prompt prints right below it). Returns True for
    rematch, False for quit."""
    curses.endwin()
    cols = shutil.get_terminal_size().columns
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'aryna.txt')
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()
    except OSError:
        lines = []
    print()
    for ln in lines:
        print(ln[:cols - 1])       # cut horizontally, never wrap
        time.sleep(PRIZE_DELAY)
    print()
    print(f"You win!  Final {score[0]}:{score[1]}")
    while True:
        ans = input("R = rematch, Q = quit > ").strip().lower()
        if ans in ('r', 'q'):
            return ans == 'r'


def reenter_curses(stdscr):
    """Fresh curses setup after the win screen ended it (rematch)."""
    stdscr = curses.initscr()
    curses.cbreak()
    curses.noecho()
    stdscr.keypad(True)
    hide_cursor(stdscr)
    stdscr.nodelay(True)
    stdscr.timeout(80)
    return stdscr

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

# Global pace dial: PONG_SPEED overrides; otherwise speed scales with
# the fitted field width (60 = design width), so narrow terminals
# (phones) keep the same felt pace instead of a faster game.
_USER_SPEED = os.environ.get('PONG_SPEED')


_speed_mult = 1.0     # runtime +/- adjustment on top of the default


def speed_step(d):
    """Multiply the pace by d (clamped 0.4..2.5). Bound to +/- keys."""
    global _speed_mult
    _speed_mult = max(0.4, min(2.5, _speed_mult * d))


def speed_factor():
    """PONG_SPEED (if valid) replaces the width default; the runtime
    +/- multiplier always applies on top."""
    base = WIDTH / 60.0
    if _USER_SPEED is not None:
        try:
            base = float(_USER_SPEED)
        except ValueError:
            pass
    return base * _speed_mult


# numpad +/- (PDCurses/windows-curses codes; None on ncurses, where
# numpad operators arrive as ESC O k / ESC O m sequences instead)
PAD_PLUS = getattr(curses, 'PADPLUS', None)
PAD_MINUS = getattr(curses, 'PADMINUS', None)


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
        # exact match with the drawn cells: paddle occupies screen rows
        # int(y)+1 .. int(y)+PADDLE_H, ball is drawn at round(by)+1 —
        # a ball that renders outside the paddle is a real miss
        return int(self.y) <= round(by) <= int(self.y) + PADDLE_H - 1


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
    ai_serve_at = time.time() + 1.2           # AI serves ~1.2s after reset
    paused = False                            # SPACE toggles (ball in play)
    last = time.time()

    while True:
        now = time.time()
        # global pace dial: PONG_SPEED, or width-based default
        dt = min(now - last, 0.2) * speed_factor()
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
            if k == 27:
                # Esc, or the first byte of a numpad escape sequence:
                # ESC O k = numpad '+', ESC O m = numpad '-'
                k2 = stdscr.getch()
                if k2 == ord('O'):
                    k3 = stdscr.getch()
                    if k3 == ord('k'):
                        speed_step(1.25)
                    elif k3 == ord('m'):
                        speed_step(0.8)
                    continue
                if k2 != -1:
                    continue        # other escape sequence: swallow it
                return False        # plain Esc quits
            if k == ord('q'):
                return False
            if k in (ord('w'), ord('W'), ord('g'), ord('G'),
                     curses.KEY_UP):
                # impulse scales with game speed: one click glides
                # IMPULSE/FRICTION * factor cells (fine nudges in slow mode)
                p1.push(-IMPULSE * speed_factor())
            elif k in (ord('s'), ord('S'), ord('v'), ord('V'),
                     curses.KEY_DOWN):
                p1.push(IMPULSE * speed_factor())
            elif k in (ord('+'), ord('=')) or k == PAD_PLUS:
                speed_step(1.25)          # faster (main or numpad)
            elif k in (ord('-'), ord('_')) or k == PAD_MINUS:
                speed_step(0.8)           # slower (main or numpad)
            elif k in (ord(' '), 10):
                if ball.waiting and server == 0:
                    ball.serve()
                else:
                    paused = not paused   # SPACE pauses / resumes

        # Paused: dt = 0 holds the paddles; the ball step is skipped
        # below and the AI serve stays deferred until resume.
        if paused:
            dt = 0.0
            ai_serve_at = time.time() + 1.2

        # AI: accelerate toward where it wants to be
        want = ball.y if ball.vx > 0 else HEIGHT / 2
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

        if ball.waiting or paused:
            scorer = None
        else:
            event = ball.step(dt, p1, p2)
            if event == 'paddle':
                beep('paddle')
            elif event == 'wall':
                beep('wall')
            scorer = 0 if event == 'score' and ball.vx > 0 else \
                     1 if event == 'score' else None
        if scorer is not None:
            score[scorer] += 1
            # Game to 11, win by 2 (official rules)
            beep('score')
            if score[scorer] >= 11 and score[scorer] - score[1 - scorer] >= 2:
                if scorer == 0:
                    if win_screen(score):   # prize into scrollback, stays
                        return True
                    return False
                stdscr.nodelay(False)
                stdscr.erase()
                x2 = max(0, WIDTH // 2 - 22)
                stdscr.addstr(HEIGHT // 2, x2, "AI wins!")
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
            ai_serve_at = time.time() + 1.2
            time.sleep(0.4)
            last = time.time()

        # Draw
        stdscr.erase()
        srv = "You (SPACE)" if server == 0 else "AI"
        hint = "  SPACE to serve" if ball.waiting and server == 0 else ""
        # speed right after the score: never truncated off the header
        spd = f" {speed_factor():.2f}x"
        pause = " PAUSED" if paused else ""
        stdscr.addstr(0, 0, (f" You {score[0]}:{score[1]} AI{spd}{pause}  "
                             f"serve: {srv}  (W/S/SPC/+/-, q=quit){hint}"
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
        stdscr.refresh()


def main(stdscr):
    hide_cursor(stdscr)
    stdscr.keypad(True)
    stdscr.nodelay(True)
    stdscr.timeout(80)
    while True:
        again = play_game(stdscr)
        if not again:
            break
        stdscr = reenter_curses(stdscr)   # win screen ended curses


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
