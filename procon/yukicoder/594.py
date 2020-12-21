"""
・自力AC！
・インタラクティブ、三分探索
・戻り値がタプル返すの忘れててWAしてた。。インタラクティブはデバッグしづらくて大変。。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def trisearch_min(lo, hi, func):
    """ 三分探索 """

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

def check1(m):
    print('? {0} 0 0'.format(m), flush=1)
    return INT()

def check2(m):
    print('? 0 {0} 0'.format(m), flush=1)
    return INT()

def check3(m):
    print('? 0 0 {0}'.format(m), flush=1)
    return INT()

x1, x2 = trisearch_min(-107, 107, check1)
x = x1 if check1(x1) <= check1(x2) else x2
y1, y2 = trisearch_min(-107, 107, check2)
y = y1 if check2(y1) <= check2(y2) else y2
z1, z2 = trisearch_min(-107, 107, check3)
z = z1 if check3(z1) <= check3(z2) else z2
print('! {0} {1} {2}'.format(x, y, z), flush=1)
