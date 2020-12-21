# -*- coding: utf-8 -*-

"""
・500点自力AC！
"""

import sys
from bisect import bisect_left, bisect_right

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
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

W,H=MAP()
P=[INT() for i in range(W)]
Q=[INT() for i in range(H)]

P.sort()
Q.sort()

ans=0
for i in range(W):
    # 反対軸を見て、自分がその中で小さい程たくさん必要になる
    cnt=H+1-bisect_left(Q, P[i])
    ans+=cnt*P[i]
for i in range(H):
    # 同値を両方で数えないように、境界の扱いを変えておく
    cnt=W+1-bisect_right(P, Q[i])
    ans+=cnt*Q[i]
print(ans)
