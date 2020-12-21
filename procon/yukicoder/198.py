"""
・自力AC
・三分探索
・最初、中央値を目指すのが最善と思ったけど、手持ち数の上限があるから、必ずしも中央値にできるとは限らない。
　その辺も条件つければいけるとは思ったけど、今回は境界値付近で同値になることもなさそうだし、
　三分探索でいけると思ったのでそっちにした。
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
    return m1, m2

def calc(m):
    res = 0
    diff = 0
    for a in A:
        diff += m - a
        res += abs(m - a)
    if diff <= b:
        return res
    else:
        return INF

b = INT()
N = INT()
A = LIST(N)

m1, m2 = trisearch_min(-1, 10**9+1, calc)
ans = min(calc(m1), calc(m2))
print(ans)
