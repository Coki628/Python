# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・N進数系
・aをdivmodした時にd=1,m=7になる数が答え。最小の、って書いてあるけど一意だねこれ多分。
・何個か試すと、基本的にa-7が常に正解、7進数以下はm=7を作れないので無理、となる。
　サンプルに優しく8進数の場合が置いてあるので、それ未満はNGと分かる。
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
EPS = 10 ** -10

a = INT()

if a >= 15:
    ans = a - 7
else:
    ans = -1
print(ans)
