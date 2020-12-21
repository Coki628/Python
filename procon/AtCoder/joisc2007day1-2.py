# -*- coding: utf-8 -*-

"""
参考：http://kagamiz.hatenablog.com/entry/2013/03/15/183614
・ほぼ自力実装なんだけど、自信なくて途中で解法調べてしまった。。
・方針は合ってることが分かったので、もうしばらく頑張って詰めたらうまくまとまった。
・グリッド、累積和、いもす、DAG、トポロジカルソート
・まず、それぞれの色について、存在しうる最大位置を調べておく。
・次に、各マスにその色があるかをO(1)で出したいので、
　各色毎に、行列独立に累積和を取っておく。
　(全マスまともにやると、配列がN*H*Wになるのでまずメモリで死ぬと思う。)
・各マスについて、一番上にある色に向かって、他の色から依存性を付ける。(有向辺を張る。)
・出来上がったグラフにトポロジカルソートかければ終わり。長かった。。
・計算量は辺を張る所が3重ループなのでO(N*H*W)=1000万。
　pythonTLE,pypyAC0.4秒。メモリ危なかった。。(64MB制限)
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def topological_sort(N: int, edges: list) -> list:
    """ トポロジカルソート(頂点数、辺集合(DAG, 0-indexed)) """

    # ここからトポロジカルソート準備
    incnts = [0] * N
    outnodes = [[] for i in range(N)]
    for i in range(len(edges)):
        # 流入するノード数
        incnts[edges[i][1]] += 1
        # 流出先ノードのリスト
        outnodes[edges[i][0]].append(edges[i][1])
    # 流入ノード数が0であるノードのセットS
    S = set()
    for i in range(N):
        if incnts[i] == 0:
            S.add(i)

    # ここからトポロジカルソート
    L = []
    # 暫定セットが空になるまでループ
    while S:
        # 暫定セットから結果リストへ1つ入れる
        L.append(S.pop())
        # 確定させたノードから流出するノードでループ
        for node in outnodes[L[-1]]:
            # 流入ノード数を1減らす
            incnts[node] -= 1
            # 流入ノードが0なら暫定セットへ
            if incnts[node] == 0:
                S.add(node)
    # 閉路があって正しくソート出来なかった場合
    if len(L) != N:
        return []
    # ソートされた頂点のリストを返却
    return L

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

N, W, H = MAP()
grid = build_grid(H, W, 0, int, padding=1)

# 各色紙の下端、右端位置
D, R = {}, {}
for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j]:
            D[grid[i][j]] = max(D.get(grid[i][j], 0), i)
            R[grid[i][j]] = max(R.get(grid[i][j], 0), j)
# 各色紙の上端、左端位置
U, L = {}, {}
for i in range(H, 0, -1):
    for j in range(W, 0, -1):
        if grid[i][j]:
            U[grid[i][j]] = min(U.get(grid[i][j], INF), i)
            L[grid[i][j]] = min(L.get(grid[i][j], INF), j)

# 行列独立に累積和(いもす)
accrow = list2d(N+1, H+2, 0)
acccol = list2d(N+1, W+2, 0)
for i in range(1, N+1):
    if i in U:
        accrow[i][U[i]] += 1
        accrow[i][D[i]+1] -= 1
    if i in L:
        acccol[i][L[i]] += 1
        acccol[i][R[i]+1] -= 1
for i in range(1, N+1):
    accrow[i] = list(accumulate(accrow[i]))
    acccol[i] = list(accumulate(acccol[i]))

edges = set()
for i in range(1, H+1):
    for j in range(1, W+1):
        # このマスで一番上になっている色
        par = grid[i][j]
        if not par:
            continue
        for k in range(1, N+1):
            # このマスにある他の全ての色から辺を張る
            if par != k and accrow[k][i] and acccol[k][j]:
                edges.add((k-1, par-1))

# トポロジカルソート
res = topological_sort(N, list(edges))
ans = [a+1 for a in res]
print(*ans)
