# -*- coding: utf-8 -*-

"""
・自力WA
・このDAGだと、直系の子孫との関係しか確かではなくて、
　全部の頂点との大小関係が必要だから、結局双方向に繋ぐが正解だった。
・WAは分かるんだけど、実際提出するとTLEとREも出す。
　REは再帰制限ぽくて、TLEも帰ってこれてないからな気がするから、
　メモとかどっかバグってんだろうけど分からん。
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
words = set()
LMS = []
for i in range(N):
    lg, m, sm = input().split()
    m = int(m)
    words.add(lg)
    words.add(sm)
    LMS.append((lg, m, sm))

# 単位と頂点番号のマッピング
tonum = {}
tostr = {}
for i, w in enumerate(list(words)):
    tonum[w] = i
    tostr[i] = w

# ここからNは頂点数
N = len(words)
# 小さい単位へ向かう有向グラフ(DAG)を構築する
nodes = [[] for i in range(N)]
for lg, m, sm in LMS:
    u = tonum[lg]
    v = tonum[sm]
    nodes[u].append((v, m))

# メモ化再帰
memo = [()] * N
def rec(u):
    if memo[u]:
        return memo[u]
    res = (1, u)
    for v, c in nodes[u]:
        tmp = rec(v)
        if tmp[0]*c > res[0]:
            # メモにはコスト最大値と一緒に、終端の頂点を持たせる
            res = (tmp[0]*c, tmp[1])
    memo[u] = res
    return res

# (始点、最大コスト、終点)
ans = (-1, -INF, -1)
for u in range(N):
    tmp = rec(u)
    if tmp[0] > ans[1]:
        ans = (u, tmp[0], tmp[1])
u, c, v = ans
print('1{0}={1}{2}'.format(tostr[u], c, tostr[v]))
