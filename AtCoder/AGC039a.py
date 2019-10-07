# -*- coding: utf-8 -*-

"""
・300点自力AC！
・ランレングス圧縮
・正しく場合分け、WA取れないとつらい系だけど何とか普通に通って良かった。
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
INF = float('inf')
MOD = 10 ** 9 + 7

S = input()
N = len(S)
K = INT()

A = []
cur = S[0]
cnt = 1
for i in range(1, N):
    if S[i-1] == S[i]:
        cnt += 1
    else:
        A.append((S[i-1], cnt))
        cnt = 1
A.append((S[N-1], cnt))

if len(A) == 1:
    print(N*K//2)
    exit()

change = 0
for s, cnt in A:
    change += cnt // 2

if S[0] == S[N-1] and A[-1][1] % 2 == 1 and A[0][1] % 2 == 1:
    print((change+1)*K-1)
else:
    print(change*K)
