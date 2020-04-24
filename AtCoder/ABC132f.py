# -*- coding: utf-8 -*-

"""
・600点自力AC！黄diffでこれは嬉しい。
・値としてN=10^9まであっても、後半はほとんど同じ数にしか遷移できなくなるので、
　それを利用して状態をまとめる。するとだいたい√Nくらいになる。
・数列の要素K個でK-1回ペアを組む。ここでも累積和で遷移がまとまる。
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

N, K = MAP()

# まず初回ペアを組ませた通り数の状態を作る
i = 1
x = N
li1 = []
li2 = []
mul = []
while i < x:
    # 前半は普通にやる
    li1.append(x)
    i += 1
    prev = x
    x = N // i
    # 後半は遷移先が同じものをまとめる
    li2.append((prev - x) * (i - 1))
    mul.append(prev - x)
li = []
if i == x:
    li += li1 + [x] + li2[::-1]
else:
    li += li1 + li2[::-1]

M = len(li)
# 後半部分はまとめてある分、係数が付く
mul = [1] * (M-len(li2)) + mul[::-1]
# dp[i][j] := i+1回ペアを組んだ時、右端がjとなる通り数
dp = list2d(K-1, M, 0)
for i in range(M):
    dp[0][i] = li[i]

for i in range(1, K-1):
    # 累積和でさらに遷移をまとめる
    acc = list(accumulate(dp[i-1]))[::-1]
    for j in range(M):
        dp[i][j] = acc[j] * mul[j] % MOD
print(sum(dp[K-2]) % MOD)
