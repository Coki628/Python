# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC
・2次元累積和
・なんとなく方針覚えてた。まあそれでもちゃんと正しく解けて良かったとは思う。
"""

import sys
from itertools import accumulate

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

N, M, Q = MAP()

# 2次元累積和
# acc[i][j] := 都市iで区間開始前で都市jで区間終了後である列車の数
acc = list2d(N, N, 0)
for i in range(M):
    l, r = MAP()
    l -= 1; r -= 1
    acc[l][r] += 1
# 区間開始前かどうかを持つ1次元目は後ろから累積和する
for j in range(N):
    for i in range(N-2, -1, -1):
        acc[i][j] += acc[i+1][j]
for i in range(N):
    acc[i] = list(accumulate(acc[i]))

ans = [''] * Q
for i in range(Q):
    p, q = MAP()
    p -= 1; q -= 1
    ans[i] = str(acc[p][q])
print('\n'.join(ans))
