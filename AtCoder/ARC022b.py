# -*- coding: utf-8 -*-

"""
・自力AC
・条件1個見落としててWAした。(sのうち最大を取る)
　最終的に適当なテストケースで原因見つけられて良かった。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

ans = s = 0
d = {}
for i, a in enumerate(A):
    ans = max(ans, i-s)
    # この区間にaが含まれているかどうか
    if a in d:
        # 区間の開始位置を前回のaの次(のうち一番右)からにする
        s = max(s, d[a] + 1)
    # aがあった位置を保持
    d[a] = i
ans = max(ans, N-s)

print(ans)
