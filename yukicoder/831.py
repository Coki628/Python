"""
・方針立たないのでとりあえず実験して性質を探った。
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

N = INT()

ans = []
mn = INF
for perm in permutations(range(1, N+1)):
    res = 0
    for i in range(N):
        a, b = perm[i], perm[(i+1)%N]
        res += a * b + (b - a)
    if res < mn:
        mn = res
        ans = [perm]
    elif res == mn:
        ans.append(perm)
print(mn)
print(len(ans))
for a in ans:
    print(*a)
