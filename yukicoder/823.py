"""
・方針出ないのでとりあえず愚直実験。何の手がかりも見つからず。。
"""

import sys
from itertools import permutations

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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

# N, K = MAP()

mat = list2d(8, 8, -1)

for N in range(1, 8):
    for K in range(1, N+1):
        # print(N, K)
        ans = 0
        for perm in permutations(range(1, N+1), K):
            A = [1] * (N+1)
            for p in perm:
                A[p-1] += A[p]
                A[p] = 0
            res = 0
            for i in range(0, N+1):
                if A[i]:
                    res += i
            # print(res)
            ans += res
        mat[N][K] = ans

for row in mat:
    print(*row)
