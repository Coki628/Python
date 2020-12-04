"""
・自力AC
・順列、グリッド、構築
・縦に1つ、横に2つずつ進めながら番号を入れていけばよしなに仕上がる。
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

N = INT()
ans = list2d(N, N, 0)
k = 0
for a in range(1, N+1):
    i = j = k
    for _ in range(N):
        ans[i%N][j%N] = a
        i += 1
        j += 2
    k += 1
for a in ans:
    print(*a)
