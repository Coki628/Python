# -*- coding: utf-8 -*-

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, X, Y, K = MAP()
A = LIST()

mod = X + Y
A2 = [0] * N
for i in range(N):
    A2[i] = A[i] % mod if A[i] % mod != 0 else mod

B = []
ans = 0
for a in A2:
    if a > X:
        B.append(a - X)
    else:
        ans += 1
B.sort()

for b in B:
    need = ceil(b, X)
    if need <= K:
        ans += 1
        K -= need
    else:
        break
print(ans)
