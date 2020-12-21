# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/iroha2019-day2/editorial-G.pdf
・頂点拡張ダイクストラ
・準備段階では拡張しないで、ダイクストラの最中に遷移する時拡張するようにした。
・ループのネストは減ったけど、これでもTLE。(無論pypyでも)
・C++で書き直してAC、0.1秒くらいで。
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

# ダイクストラ
def dijkstra(N: int, nodes: list, src: int) -> list:
    res = list2d(K*2, N, INF)
    que = [(0, 0, src)]
    res[0][src] = 0
    while len(que) != 0:
        dist, k, cur = heappop(que)
        x,y=XY[cur]
        for nxt, cost in nodes[cur]:
            if res[k][cur] + cost < res[k][nxt]:
                res[k][nxt] = res[k][cur] + cost
                heappush(que, (res[k][nxt], k, nxt))
            # 花を買う分のK方向の移動をする辺を追加する
            idx=0
            while x!=0 and k+idx*x<K:
                idx+=1
                nxtk=k+idx*x
                cost2=cost+idx*y
                if res[k][cur] + cost2 < res[nxtk][nxt]:
                    res[nxtk][nxt] = res[k][cur] + cost2
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

res=dijkstra(N, nodes, 0)
ans=INF
# 花をK本以上買った頂点Nから最小値を取る
for k in range(K, K*2):
    ans=min(ans, res[k][N-1])
if ans!=INF:
    print(ans)
else:
    print(-1)
