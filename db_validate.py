# Validates the endgame tablebase through the engine: known-result positions
# must come back with database win/draw scores.
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'build', 'lib.win-amd64-cpython-314')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'python')))
import search_engine as se

def bit(row, col):
    return 1 << (row * 8 + col)

# 2 kings vs 1 king, side with 2 kings to move: forced win
p1k = bit(7, 0) | bit(5, 2)
p2k = bit(0, 1)
r = se.search_position(0, 0, p1k, p2k, 1, 0.2, 50)
print("2K vs 1K (p1 to move):  eval =", r[1][4], " (expect > 900)")

# same position from the lone king's side: forced loss
r = se.search_position(0, 0, p1k, p2k, 2, 0.2, 50)
print("1K vs 2K (p2 to move):  eval =", r[1][4], " (expect < -900)")

# 1 king vs 1 king, far apart: draw
p1k = bit(7, 0)
p2k = bit(0, 1)
r = se.search_position(0, 0, p1k, p2k, 1, 0.2, 50)
print("1K vs 1K:               eval =", r[1][4], " (expect 0)")

# 3 kings vs 2 kings: win for the 3
p1k = bit(7, 0) | bit(5, 2) | bit(5, 4)
p2k = bit(0, 1) | bit(2, 1)
r = se.search_position(0, 0, p1k, p2k, 1, 0.2, 50)
print("3K vs 2K (p1 to move):  eval =", r[1][4], " (expect > 900)")

# man about to promote vs distant king: should at least not be a loss
p1 = bit(1, 2)
p2k = bit(6, 1)
r = se.search_position(p1, 0, 0, p2k, 1, 0.2, 50)
print("1 man near prom vs 1K:  eval =", r[1][4])
