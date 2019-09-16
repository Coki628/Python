# -*- coding: utf-8 -*-

"""
・試しにローリングハッシュしないで普通に文字列のまんまやる版
・3.0→1.8秒と速くなってしまった。ロリハする意味って。。
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

T = input()
P = input()
N = len(T)
M = len(P)

ans = []
p = P
for i in range(N-M+1):
    # Tの区間[i,i+M)の文字列を取得
    t = T[i:i+M]
    if t == p:
        ans.append(i)

if len(ans):
    [print(i) for i in ans]
