"""
参考：https://yukicoder.me/problems/no/999/editorial
・公式解
・前計算、累積和
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
EPS = 10 ** -10

N = INT()
NN = N * 2
A = LIST()

# 左からi個取った場合と右からi個取った場合を前計算する
accl = []
accr = []
for i in range(0, NN, 2):
    accl.append(A[i] - A[i+1])
    accr.append(A[i+1] - A[i])
accl = [0] + list(accumulate(accl))
accr = list(accumulate(accr[::-1]))[::-1] + [0]
ans = -INF
for i in range(N+1):
    ans = max(ans, accl[i] + accr[i])
print(ans)
