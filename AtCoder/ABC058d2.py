# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC
・全区間総和(言い方合ってるかな？)、縦横独立。
・印象的だったからやり方は覚えてたけど、久々に見てちゃんとソラで書けたからよかった。
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

N, M = MAP()
A = LIST()
B = LIST()

# データの持ち方を座標から区間長にして累積和
A2 = []
for i in range(N-1):
    A2.append(A[i+1] - A[i])
B2 = []
for i in range(M-1):
    B2.append(B[i+1] - B[i])
A2 = [0] + list(accumulate(A2))
B2 = [0] + list(accumulate(B2))

# 縦横独立に、累積和から全区間総和を取る
lsm = rsm = 0
for i in range(N):
    lsm += A2[i] * (N-i-1)
    rsm += A2[i] * i
    lsm %= MOD
    rsm %= MOD
h = rsm - lsm
lsm = rsm = 0
for i in range(M):
    lsm += B2[i] * (M-i-1)
    rsm += B2[i] * i
    lsm %= MOD
    rsm %= MOD
w = rsm - lsm

ans = h * w % MOD
print(ans)
