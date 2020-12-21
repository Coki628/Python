"""
・自力AC
・好きに操作できるので下限lをK回やるのと上限rをK回やるのの間は全部網羅できるってやつ。
・後はその範囲内にMの倍数があるかどうか。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

l, r, M, K = MAP()

mn = l * K
mx = r * K
l2 = mn if mn % M == 0 else mn - mn%M + M
r2 = mx - mx%M
if l2 <= r2:
    Yes()
else:
    No()
