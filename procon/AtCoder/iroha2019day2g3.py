# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/iroha2019-day2/editorial-G.pdf
・頂点拡張ダイクストラ
・速度対策で隣接リストを1次元のままにしてみたけどpypyでもTLE、てかむしろ遅くなってた。。
・まあ、K*MlogNだと1000*2000logN=200万logNだからどっちにしてもダイクストラきついか…。
"""

import sys
from heapq import heappop, heappush

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

def dijkstra(N: int, nodes: list, src: int) -> list:
    """ ダイクストラ(頂点数, 隣接リスト(0-indexed), 始点) """

    # 頂点[ある始点からの最短距離] (経路自体を知りたい時はここに前の頂点も持たせる)
    res = [INF] * N
    # スタート位置
    que = [(0, src)]
    res[src] = 0
    # キューが空になるまで
    while len(que) != 0:
        # srcからの距離, 現在のノード
        dist, cur = heappop(que)
        # 出発ノードcurの到着ノードでループ
        for nxt, cost in nodes[cur]:
            # 今回の経路のが短い時
            if res[cur] + cost < res[nxt]:
                res[nxt] = res[cur] + cost
                # 現在の移動距離をキューの優先度として、早い方から先に処理するようにする
                heappush(que, (res[nxt], nxt))
    # ノードsrcからの最短距離リストを返却
    return res

N,M,K=MAP()
nodes=[[] for i in range(N)]
for i in range(M):
    a,b,c=MAP()
    nodes[a-1].append((b-1, c))
    nodes[b-1].append((a-1, c))
# 最後の頂点はK方向だけの遷移をさせたいので自己ループを追加しておく
nodes[N-1].append((N-1, 0))
XY=[]
for i in range(N):
    x,y=MAP()
    XY.append((x, y))

# K方向とN方向の遷移をするための隣接リスト(2方向だけど速度のために1次元で作る)
extended=[[] for u in range(N*K*2)]
for k in range(K*2):
    for u in range(N):
        x,y=XY[u]
        for v,c in nodes[u]:
            # 元々のN方向の移動
            extended[k*N+u].append((k*N+v, c))
            # 花を買う分のK方向の移動をする辺を追加する
            idx=0
            while x!=0 and k+idx*x<K:
                idx+=1
                extended[k*N+u].append(((k+idx*x)*N+v, c+idx*y))

res=dijkstra(N*K*2, extended, 0)
ans=INF
# 花をK本以上買った頂点Nから最小値を取る
for k in range(K, K*2):
    ans=min(ans, res[k*N+(N-1)])
if ans!=INF:
    print(ans)
else:
    print(-1)
