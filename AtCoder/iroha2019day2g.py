# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/iroha2019-day2/editorial-G.pdf
・頂点拡張ダイクストラ
・準備段階で全ての頂点を拡張して隣接リストに詰めたら盛大にTLE
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

# ダイクストラ(nodes:頂点を2次元で持った隣接リスト)
def dijkstra(N: int, nodes: list, src: int) -> list:
    res = list2d(K*2, N, INF)
    que = [(0, 0, src)]
    res[0][src] = 0
    while len(que) != 0:
        dist, k, cur = heappop(que)
        for nxtk, nxt, cost in nodes[k][cur]:
            if res[k][cur] + cost < res[nxtk][nxt]:
                res[nxtk][nxt] = res[k][cur] + cost
                heappush(que, (res[nxtk][nxt], nxtk, nxt))
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

# K方向とN方向の遷移をするための2次元の隣接リスト
extended=[[[] for u in range(N)] for k in range(K*2)]
for k in range(K*2):
    for u in range(N):
        x,y=XY[u]
        for v,c in nodes[u]:
            # 元々のN方向の移動
            extended[k][u].append((k, v, c))
            # 花を買う分のK方向の移動をする辺を追加する
            idx=0
            while x!=0 and k+idx*x<K:
                idx+=1
                extended[k][u].append((k+idx*x, v, c+idx*y))

res=dijkstra(N, extended, 0)
ans=INF
# 花をK本以上買った頂点Nから最小値を取る
for k in range(K, K*2):
    ans=min(ans, res[k][N-1])
if ans!=INF:
    print(ans)
else:
    print(-1)
