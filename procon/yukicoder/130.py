# -*- coding: utf-8 -*-

"""
・cf1285dで出たものとほぼ同じ。てかこっちが先みたいだけど。
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

N = INT()
A = LIST()

def rec(li, k):
    if not li or k < 0:
        return 0
    li1, li2 = [], []
    # 今見ている桁が1の要素と0の要素で振り分ける
    for a in li:
        if a & 1<<k:
            li1.append(a)
        else:
            li2.append(a)
    # 全て片方にあれば、この桁は(最大値になりうる)全要素で0にできる
    if not li1 or not li2:
        return rec(li, k - 1)
    else:
        # 0も1もあれば、この桁が(最大値になりうる)いずれかの要素で立つのは確定で、どちらを立たせた方が得かを下の桁に調べに行く
        return min(rec(li1, k - 1), rec(li2, k - 1)) + (1 << k)

print(rec(A, 30))
