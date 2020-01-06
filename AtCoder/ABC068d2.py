# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC！
・構築系、最後はシミュレーション。
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

K = INT()

N = 50
d, m = divmod(K, N)
# 初期位置を決める
base = N - 1 + d
A = [base] * N
# 残りは逆操作をシミュレーション
for i in range(m):
    k = A.index(min(A))
    for j in range(N):
        if j == k:
            A[j] += N
        else:
            A[j] -= 1

print(len(A))
print(*A)
