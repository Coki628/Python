# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/code-festival-2017-quala/editorial.pdf
・200点自力ならず。これは残念。
・規則性を探してひたすら悩んでしまった。
・式にできる。後はボタン押した回数全探索。
"""

import sys

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

N,M,K=MAP()

for i in range(N+1):
    for j in range(M+1):
        if i*(M-j)+j*(N-i)==K:
            Yes()
            exit()
No()
