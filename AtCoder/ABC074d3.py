# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・手こずったけど自力AC！
・ワーシャルフロイド応用
・通せたからいいけど、当時のがスッキリまとめててへこむわ。。
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

def warshall_floyd(N: int, graph: list) -> list:
    """ ワーシャルフロイド(頂点数, 隣接行列(0-indexed)) """
    from copy import deepcopy

    res = deepcopy(graph)
    for i in range(N):
        # 始点 = 終点、は予め距離0にしておく
        res[i][i] = 0
    # 全頂点の最短距離
    for k in range(N):
        for i in range(N):
            for j in range(N):
                res[i][j] = min(res[i][j], res[i][k] + res[k][j])
    # 負の閉路(いくらでもコストを減らせてしまう場所)がないかチェックする
    for i in range(N):
        if res[i][i] < 0:
            return []
    return res

def build_grid(H, W, intv, _type, space=True, padding=False):
    # 入力がスペース区切りかどうか
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
    # 余白の有無
    if padding:
        offset = 1
    else:
        offset = 0
    grid = list2d(H+offset*2, W+offset*2, intv)
    for i in range(offset, H+offset):
        row = _list()
        for j in range(offset, W+offset):
            grid[i][j] = row[j-offset]
    return grid

def check1(A, B):
    for i in range(N):
        for j in range(N):
            # 1つでも最短経路が更新されていればNG
            if A[i][j] != B[i][j]:
                return False
    return True

def check2(G):
    res = 0
    S = set()
    for k in range(N):
        for i in range(N):
            for j in range(N):
                # 3頂点が異なってかつ、経由のが近い場合は直接の辺(i,j)は不要
                if i != j and j != k and k != i \
                        and G[i][k] + G[k][j] == G[i][j]:
                    # 同じペアが何回も該当する場合もあるので1回だけ処理する
                    if (i, j) not in S:
                        S.add((i, j))
                        res += G[i][j]
    return res

N = INT()
G = build_grid(N, N, 0, int)

# 妥当性の確認
wf = warshall_floyd(N, G)
if not check1(G, wf):
    print(-1)
    exit()

# 全体から不要な辺のコストを引く
total = 0
for i in range(N):
    for j in range(N):
        total += wf[i][j]
ans = (total - check2(wf)) // 2
print(ans)
