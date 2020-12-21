# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc132/editorial.pdf
　　　https://www.youtube.com/watch?v=mso8tE1yMl8
・頂点拡張
・この前やったばっかのテーマだったのに解けなかったのはつらい。
・重みなしグラフなので、公式解通りBFSでもやってみる。
"""

import sys
from collections import deque

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
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

def bfs(N: int, nodes: list, src: int) -> list:
    """ 幅優先探索(頂点数, 隣接リスト(0-indexed), 始点) """

    # 頂点[ある始点からの最短距離] (経路自体を知りたい時はここに前の頂点も持たせる)
    res = [[INF] * N for i in range(3)]
    # スタート位置
    que = deque([(0, src)])
    res[0][src] = 0
    # キューが空になるまで
    while len(que) != 0:
        # 現在のノード
        k, cur = que.popleft()
        # 出発ノードcurの到着ノードでループ
        for nxtk, nxt in nodes[k][cur]:
            # 今回の経路のが短い時
            if res[k][cur] + 1 < res[nxtk][nxt]:
                res[nxtk][nxt] = res[k][cur] + 1
                # 後ろに詰めて、近い方から処理するようにする
                que.append((nxtk, nxt))
    # ノードsrcからの最短距離リストを返却
    return res

N,M=MAP()
nodes=[[] for i in range(N)]
for i in range(M):
    u,v=MAP()
    nodes[u-1].append(v-1)
S,T=MAP()
S-=1
T-=1

# 元の頂点を3層に拡張する
nodes2=[[[] for j in range(N)] for i in range(3)]
for i in range(3):
    for u, node in enumerate(nodes):
        for v in node:
            nodes2[i][u].append(((i+1)%3, v))

res=bfs(N, nodes2, S)
if res[0][T]==INF:
    print(-1)
else:
    # けんけんぱの回数なので/3する
    print(res[0][T]//3)
