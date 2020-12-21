"""
・Shortest用
"""

import sys
import gc
from math import hypot

def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())

N = INT()
M = INT()
D = INT()
K = INT()
cur = []
x, y = MAP()
cur.append((x, y))
XY = set()
for _ in range(N-1):
    x, y = MAP()
    XY.add((x, y))

cnt = [0] * (K+1)
cnt[0] = 1
for k in range(1, K+1):
    nxt = []
    for x, y in cur:
        for i in range(x-D, x+D+1):
            for j in range(y-D, y+D+1):
                if (i, j) in XY and hypot(x-i, y-j) <= D:
                    cnt[k] += 1
                    nxt.append((i, j))
                    XY.remove((i, j))
    cur = nxt
    if k % 5 == 0:
        gc.collect()
ans = sum(cnt[-M:])
print(ans)
