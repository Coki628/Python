# -*- coding: utf-8 -*-

"""
・自力AC！
・木の走査
・tkppc4day2bと同じ要領で、必要な情報を戻り値に持たせて集めていく。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, X = MAP()
X -= 1
A = LIST()
nodes = [[] for i in range(N)]
for i in range(N-1):
    a, b = MAP()
    nodes[a-1].append(b-1)
    nodes[b-1].append(a-1)

def rec(cur, prev):
    res = 0
    for nxt in nodes[cur]:
        if nxt != prev:
            res += rec(nxt, cur)
    # ここより先で宝石があった頂点までの総コスト
    if res == 0:
        # この時点で1つもなければ現在の頂点次第(最初以外)
        if prev != -1:
            res += A[cur]
        return res
    else:
        # 1つでもあれば通らないといけないので、現在の頂点分のコストを足して返却(最初以外)
        if prev != -1:
            res += 1
        return res

# 行って帰ってくるので*2
print(rec(X, -1)*2)
