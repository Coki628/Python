"""
参考：https://qiita.com/drken/items/f2ea4b58b0d21621bd51
・写像12相勉強会
・8相
・N個の玉をK個の箱に1個ずつ入れるが、箱の区別がない。
　つまり箱が足りるなら、全部同じ感じになるので1通り。
　足りないなら無理なので0通り。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N, K = MAP()

# 箱が足りてればOK、足りなければ入れられない
if N <= K:
    print(1)
else:
    print(0)
