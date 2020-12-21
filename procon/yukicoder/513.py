"""
・自力ならず。。
・インタラクティブ、三分探索
・三分探索だとクエリ数オーバーみたいだ。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def trisearch_min(lo, hi, func):
    while lo+2 < hi:
        m1 = (lo*2+hi) // 3
        m2 = (lo+hi*2) // 3
        res1 = func(m1)
        res2 = func(m2)
        if res1 <= res2:
            hi = m2
        else:
            lo = m1
    return (m1, m2)

def check1(x):
    print(x, 0, flush=1)
    res = INT()
    if res == 0:
        exit()
    else:
        return res

def check2(y):
    print(0, y, flush=1)
    res = INT()
    if res == 0:
        exit()
    else:
        return res

sx, tx = trisearch_min(-1, 100001, check1)
sy, ty = trisearch_min(-1, 100001, check2)
for x in range(sx, tx+1):
    for y in range(sy, ty+1):
        print(x, y, flush=1)
        res = INT()
        if res == 0:
            exit()
