# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/iroha2019-day2/editorial-G.pdf
・頂点拡張ダイクストラ
・scipyのライブラリでやってみる。
・速くならなかった。。。
・あと、疎行列の初期値が0で辺なしの扱いになるから、
　コスト0の辺があっても張れないっぽい。
"""

import sys
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import lil_matrix

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

# K方向とN方向の遷移をするための隣接行列
extended=lil_matrix((N*K*2, N*K*2))
for k in range(K*2):
    for u in range(N):
        x,y=XY[u]
        for v,c in nodes[u]:
            # 元々のN方向の移動
            if extended[k*N+u,k*N+v] == 0:
                extended[k*N+u,k*N+v] = c
            else:
                extended[k*N+u,k*N+v] = min(extended[k*N+u,k*N+v], c)
            # 花を買う分のK方向の移動をする辺を追加する
            idx=0
            while x!=0 and k+idx*x<K:
                idx+=1
                if extended[k*N+u,(k+idx*x)*N+v] == 0:
                    extended[k*N+u,(k+idx*x)*N+v] = c+idx*y
                else:
                    extended[k*N+u,(k+idx*x)*N+v] = min(extended[k*N+u,(k+idx*x)*N+v], c+idx*y)

res=dijkstra(extended.tocsr(), indices=0)
ans=INF
# 花をK本以上買った頂点Nから最小値を取る
for k in range(K, K*2): 
    ans=min(ans, res[k*N+(N-1)])
if ans!=INF:
    print(int(ans))
else:
    print(-1)
