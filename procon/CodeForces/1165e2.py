# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/67041
・区間和の総和
・Aの各位置の要素がそれぞれ総和で何回出現するかは、
　(i+1) * (N-i) := その要素を含む区間が取りうる(左端の数 * 右端の数) で出せる。
・Aの位置は固定なので、先にAの各要素をこの回数で重み付けしておく。
・その大きさに応じて対角線上になるようにBとペアにしていけばOK。
"""

import sys
from itertools import accumulate

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
INF = 10 ** 18
MOD = 998244353

N = INT()
A = LIST()
B = LIST()

# Aを出現回数で重み付けする
A = [a * (i+1) * (N-i) for i, a in enumerate(A)]
A.sort()
B.sort(reverse=1)
ans = 0
for i in range(N):
    ans += A[i] * B[i]
    ans %= MOD
print(ans)
