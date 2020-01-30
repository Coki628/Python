# -*- coding: utf-8 -*-

"""
・本当に色々試行錯誤の末、800点自力AC！疲れた。
・全体の方針としては、自分より前がこなさないといけない試合数に応じて、
　自分も必要な試合数を割り出して、親に返して、を優勝者(木の根)まで戻す、って感じ。
・基本的には自分の子ノードで一番試合数がかかったやつなんだけど、
　最大試合数が重複していた場合、1つ多く増やさないといけないのが厄介。
　最大でなくても重複があった場合、最大までの間に隙間があれば挟みこめるし、
　なければ最大を増やす、と状況によって対応が変わる。
・setで管理したり最大から最小までループ回したりでTLE紆余曲折した後、
　Counterしたやつをキーがある場所だけ見て、隙間がどのくらい空くかは、
　前のキーとの差分から取る、のやり方に行き着いてやっとAC。。
・解説見た。これも木DPだそうだ。やっぱり特にDP配列作る必要がなくても、
　子ノードからの情報を元に自分の値を確定して親に戻していく作業をすれば
　それが木DPと言って良さそう。
"""

import sys
from collections import Counter

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

def dfs(N, nodes, src):
    """ DFS(木、再帰、重みなし) """

    def rec(u, prev):
        C = Counter()
        for v in nodes[u]:
            if v != prev:
                C[rec(v, u) + 1] += 1
        if not C:
            return 0
        # それまでの試合数が多くかかる方から見ていく
        C = sorted(C.items(), reverse=1)
        # 元の最大試合数
        mx = C[0][0]
        # 重複による追加分
        add = C[0][1] - 1
        # 重複をカバーする空き時間
        vacant = 0
        for i, (k, v) in enumerate(C[1:], 1):
            k2, _ = C[i-1]
            # 試合がなくて他に当てられる時間を足す
            vacant += k2 - k - 1
            # 今回の重複分を引く
            vacant -= v - 1
            # 空き時間が足りなければ、最大以降に追加する
            if vacant < 0:
                add += abs(vacant)
                vacant = 0
        return mx + add

    return rec(src, -1)

N = INT()
nodes = [[] for i in range(N)]
for a in range(1, N):
    b = INT()
    b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

print(dfs(N, nodes, 0))
