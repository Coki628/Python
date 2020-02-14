# -*- coding: utf-8 -*-

"""
・なんとか自力AC。
・1を1つでも含む区間 → 全体 - 1を1つも含まない区間、の言い換えで状況整理できるやつ。
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def nH2(n):
    return (n+1)*n//2

for _ in range(INT()):
    N, M = MAP()

    cnt1 = M
    cnt0 = N - M
    total = nH2(N)
    if cnt1 >= N//2:
        print(total - cnt0)
        continue

    di, mo = divmod(cnt0, cnt1+1)
    dicnt = cnt1+1 - mo
    # なるだけ均等に分けて、端数も均等に1ずつ割り振る
    ans = total - (dicnt * nH2(di) + mo * nH2(di+1))
    print(ans)
