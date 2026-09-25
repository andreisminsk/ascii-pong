#!/usr/bin/env python3
"""ASCII Pong with paddle physics.

Keys give paddles an impulse (velocity), which decays via friction —
so paddles accelerate, glide, and coast to a stop. Ball bounce angle
depends on where it hits the paddle, and rallies speed the ball up.
"""
import curses
import math
import random
import subprocess
import time


# macOS system sounds (fall back to terminal bell elsewhere)
_SOUNDS = {
    'paddle': '/System/Library/Sounds/Tink.aiff',
    'wall':   '/System/Library/Sounds/Pop.aiff',
    'serve':  '/System/Library/Sounds/Tink.aiff',
    'score':  '/System/Library/Sounds/Submarine.aiff',
}


def beep(kind='paddle'):
    try:
        path = _SOUNDS.get(kind)
        if path:
            subprocess.Popen(['afplay', path], stdout=subprocess.DEVNULL,
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


def play_game(stdscr):
    """One game to 11. Returns True for rematch, False for quit."""
    stdscr.nodelay(True)   # restore non-blocking input after any game-end prompt
    stdscr.timeout(80)
    p1 = Paddle((HEIGHT - PADDLE_H) / 2)
    p2 = Paddle((HEIGHT - PADDLE_H) / 2, AI_MAX_SPEED)
    server = random.choice((0, 1))            # 0 = you (left), 1 = AI (right)
    ball = Ball()
    ball.reset(serve_left=(server == 0), paddle_y=(p1.y if server == 0 else p2.y))
    score = [0, 0]
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
            if k in (ord('q'), 27):
                return False
            if k in (ord('w'), ord('W'), curses.KEY_UP):
                p1.push(-IMPULSE)
            elif k in (ord('s'), ord('S'), curses.KEY_DOWN):
                p1.push(IMPULSE)
            elif k in (ord(' '), 10) and ball.waiting and server == 0:
                ball.serve()

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

        if ball.waiting:
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
                stdscr.nodelay(False)
                stdscr.erase()
                msg = "You win!" if scorer == 0 else "AI wins!"
                stdscr.addstr(HEIGHT // 2, WIDTH // 2 - 8, msg)
                stdscr.addstr(HEIGHT // 2 + 1, WIDTH // 2 - 22,
                              f"Final {score[0]}:{score[1]}   R = rematch, Q = quit")
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
        stdscr.addstr(0, 0, f" You {score[0]} : {score[1]} AI  serve: {srv}  "
                           f"(Up/Down or W/S, q=quit){hint}")
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
    while play_game(stdscr):
        pass


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
