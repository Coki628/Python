# -*- coding: utf-8 -*-

"""
・結構さくっと自力AC(終了後)。Dよりこっち先に取り組めばよかったかな。。
・累積和、2冪
・Bを1桁ずつ減らしながら同じ処理をすると、最終的に自分より左にある1の数だけ、
　自分のところを1が通過する感じになる。これは累積和しておけばすぐ取れる。
・条件はANDなので、このうちAも1の場所だけ数がカウントされる。
・後は2冪毎に何回足されるか、MOD取りながら数えていけばOK。
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

N, M = MAP()
A = [0] * max(M-N, 0) + list(map(int, input()))
B = [0] * max(N-M, 0) + list(map(int, input()))
NM = max(N, M)

acc = list(accumulate(B))
A = A[::-1]
acc = acc[::-1]
ans = 0
for i in range(NM):
    if A[i]:
        ans += pow(2, i, MOD) * acc[i]
        ans %= MOD
print(ans)
