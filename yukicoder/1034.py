"""
・自力TLE。。
・グリッド全部を見るのは無理なので、方向転換するまでをO(1)に。
・でも確かにこれだと10^5グリッドならいけそうだけど10^9グリッドは無理だね。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def go_right(h, w, lim, dist):
    dist += (lim-w)
    w = lim
    lim -= 1
    return (h, w), lim, dist

def go_down(h, w, lim, dist):
    dist += (lim-h)
    h = lim
    lim -= 1
    return (h, w), lim, dist

def go_left(h, w, lim, dist):
    dist += (w-lim)
    w = lim
    lim += 1
    return (h, w), lim, dist

def go_up(h, w, lim, dist):
    dist += (h-lim)
    h = lim
    lim += 1
    return (h, w), lim, dist

for _ in range(INT()):
    N, i, j = MAP()

    ln = N - 1
    dist = 0
    cur = (0, 0)
    rlim = N - 1
    dlim = N - 1
    llim = 0
    ulim = 1
    while 1:
        if cur[0] == i and cur[1] <= j <= rlim:
            dist += j - cur[1]
            break
        cur, rlim, dist = go_right(*cur, rlim, dist)
        if cur[1] == j and cur[0] <= i <= dlim:
            dist += i - cur[0]
            break
        cur, dlim, dist = go_down(*cur, dlim, dist)
        if cur[0] == i and llim <= j <= cur[1]:
            dist += cur[1] - j
            break
        cur, llim, dist = go_left(*cur, llim, dist)
        if cur[1] == j and ulim <= i <= cur[0]:
            dist += cur[0] - i
            break
        cur, ulim, dist = go_up(*cur, ulim, dist)
    print(dist)
