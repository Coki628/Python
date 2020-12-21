# -*- coding: utf-8 -*-

"""
・蟻本演習3-4-4
・自力WA
・lcm(a, b) = a * b / gcd(a, b) を使う方針は悪くなかった。
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

A, B, M = MAP()

A, B = min(A, B), max(A, B)

S = ''
# BCDがAになる
if B % A == 0:
    # Bだけ残る
    S += '1' * B
# GCDが1になる
else:
    # A * B
    S += str(*list(range(1, A))) + (str(A) * (B-A+1)) + str(*list(range(A-1, 0, -1)))

N = len(S)
ans = 0
for i in range(N):
    ans += int(S[i]) * pow(10, i, M)
    ans %= M
print(ans)
