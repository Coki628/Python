# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/code-festival-2015-final
・蟻本演習3-3-3
・いもす法
・公式解。こっちは要考察力って感じ。
・log乗らないし、速度は全然速い。python,pypy共にAC0.3秒。
"""

import sys
from itertools import accumulate

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

MAX = 10 ** 5
N = INT()
imos = [0] * (MAX+1)
ST = []
for i in range(N):
    s, t = MAP()
    imos[s] += 1
    imos[t] -= 1
    ST.append((s, t))
imos = list(accumulate(imos))

# 全区間が覆っている区間の左端と右端を探す
mx = max(imos)
l = INF
r = 0
for i, a in enumerate(imos):
    if a == mx:
        l = min(l, i)
        r = max(r, i)

# 上記左端から右端までを全て含むような区間があれば-1できる
for s, t in ST:
    if s <= l and r < t:
        print(mx-1)
        exit()
print(mx)
