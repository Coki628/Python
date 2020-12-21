"""
・実験用
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
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
S = input()

S2 = []
for i in range(1<<K):
    S2.append(S[i%N])

def check(a, b):
    if a == b:
        return a
    if a == 'R' and b == 'S' \
            or a == 'S' and b == 'P' \
            or a == 'P' and b == 'R':
        return a
    else:
        return b

res = [[] for i in range(K+1)]
for i in range(1<<K):
    res[0].append(S2[i])
for i in range(1, K+1):
    for j in range(0, len(res[i-1]), 2):
        a = res[i-1][j]
        b = res[i-1][j+1]
        winner = check(a, b)
        res[i].append(winner)
print()
