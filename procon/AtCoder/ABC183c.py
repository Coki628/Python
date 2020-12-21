"""
・さくっと自力AC
・順列全探索
"""

import sys
from itertools import permutations

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
G = list2d(N, N, 0)
for i in range(N):
    G[i] = LIST()

ans = 0
for perm in permutations(range(1, N)):
    perm = [0] + list(perm)
    dist = 0
    for i in range(N):
        u = perm[i]
        v = perm[(i+1)%N]
        dist += G[u][v]
    if dist == K:
        ans += 1
print(ans)
