# -*- coding: utf-8 -*-

"""
・自力AC
・式変形
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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

K, A, B = MAP()

# 上がっていかない場合
if B >= A:
    # 初回で行ければOK、そうでなければNG
    if A >= K:
        print(1)
    else:
        print(-1)
else:
    # A + (A-B)*x >= K → x >= (K-A) / (A-B)
    x = ceil((K-A), (A-B))
    # 2回に1回しか上がらないので、その調整
    print(x*2+1)
