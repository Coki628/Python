# -*- coding: utf-8 -*-

"""
参考：https://www.youtube.com/watch?v=8lm8o8L9Bmw
・300点自力ならず…。
・左右から累積GCD
・そう聞いたら解法気付いて何も見ずに解けた。
・この前似たようなやつ(tenka1_2019c)にやられたばっかなのに、
　どうして累積ピンと来なかったんだろう…。
・解説から、gcd(0, x)=xだと分かったので、番兵足して整理した。
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd

def input(): return sys.stdin.readline().strip()
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

N=INT()
A=LIST()

# 予め左右から累積GCDを取っておく(先頭と末尾は番兵の0)
L=[0]*(N+2)
for i in range(N):
    L[i+1]=gcd(L[i], A[i])
R=[0]*(N+2)
for i in range(N-1, -1, -1):
    R[i+1]=gcd(R[i+2], A[i])

# 1箇所除いてどうなるかを全部試しても、各回GCD1回ずつで済む
ans=0
for i in range(N):
    ans=max(ans, gcd(L[i], R[i+2]))
print(ans)
