# -*- coding: utf-8 -*-

"""
参考：http://arc051.contest.atcoder.jp/data/arc/051/editorial.pdf
・公式解の方針。
・先にシミュレーションする。a * X >= max(A) になったら、後は先頭が末尾へ、
　をずっと繰り返すので、まとめて計算できる。
"""

import sys
from heapq import heapify, heappop, heappush

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

N, X, Y = MAP()
A = LIST()

# 例外処理
if X == 1:
    A.sort()
    [print(a) for a in A]
    exit()

# a * X >= max(A) になるまでシミュレーション
que = A.copy()
heapify(que)
mx = max(A)
while Y > 0:
    a = heappop(que)
    if a * X >= mx:
        heappush(que, a)
        break
    a *= X
    Y -= 1
    heappush(que, a)
    mx = max(mx, a)

for i in range(N):
    a = heappop(que)
    A[i] = a
# mod個目まではdiv+1回、それ以降はdiv回、Xを掛ける
div, mod = divmod(Y, N)
for i in range(N):
    A[i] *= pow(X, div, MOD)
    if i < mod:
        A[i] *= X
    A[i] %= MOD
# div+1回やった方を後ろへ
A =  A[mod:] + A[:mod]
[print(a) for a in A]
