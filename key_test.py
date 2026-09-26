#!/usr/bin/env python3
"""Key-code probe: press keys, see what curses reports. q quits.
Run if +/- seems dead in the game:
    python3 key_test.py
Then press + and - and compare with the expected codes:
    '+' = 43, '=' = 61, '-' = 45, '_' = 95
"""
import curses


def main(stdscr):
    stdscr.keypad(True)
    stdscr.nodelay(False)
    stdscr.addstr(0, 0, "press keys to see their codes (q quits)")
    stdscr.refresh()
    while True:
        k = stdscr.getch()
        if k == -1:
            continue
        name = curses.keyname(k).decode() if k > 255 else chr(k)
        stdscr.erase()
        stdscr.addstr(0, 0, f"key code: {k:>4}   name: {name}   (q quits)")
        stdscr.refresh()
        if k in (ord('q'), 27):
            return


if __name__ == "__main__":
    curses.wrapper(main)
