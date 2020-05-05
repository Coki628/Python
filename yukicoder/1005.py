# -*- coding: utf-8 -*-

"""
・さくっと自力AC！
・連続部分文字列
・N^2が通る制約なので、部分列のスライスも特に気にせず普通にやってOK。
・ちなみに公式解説より、Tを先にする形でSと連結させれば、Z-algorithmでO(N+M)いける。
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

S = input()
T = input()
N = len(S)
M = len(T)

if M == 1:
    if S.count(T):
        print(-1)
    else:
        print(0)
    exit()

i = 0
ans = 0
while i < N-M+1:
    s = S[i:i+M]
    if s != T:
        i += 1
    else:
        ans += 1
        i += M - 1
print(ans)
