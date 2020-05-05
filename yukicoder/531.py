# -*- coding: utf-8 -*-

"""
・ゆきこ茶diff初自力ならず。。WA分かんなくてテストケース見た。
・常に1度に全員移動しか無理かと思ったけど、うまく1つ飛ばしで全体の半分を持てれば、
　関係性の辺が全て消えるので2回かけて移動が可能。
・気づかなかったわー。。。
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

N, M = MAP()

if N <= M:
    print(1)
elif N % 2 == 0 and N // 2 <= M:
    print(2)
else:
    print(-1)
