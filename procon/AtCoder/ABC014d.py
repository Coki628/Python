# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc014
　　　http://satanic0258.hatenablog.com/entry/2017/02/23/222647
　　　http://kakira.hatenablog.jp/entry/2014/09/17/122249
・LCA(最小共通祖先)
・ダブリング
・仕組みの理解は参考2番目のスライドがすごく分かりやすかった。
・vの2^(k-1)つ上の2^(k-1)つ上は2^kつ上
　→vの4つ上の4つ上は8つ上(そりゃそうだ)って具体的に考えると分かりやすい。
・計算量は10万のNlogNで、python2.0秒ギリギリ、pypy1.0秒でAC。
"""
import sys

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

N=INT()
nodes=[[] for i in range(N)]
for i in range(N-1):
    x,y=MAP()
    nodes[x-1].append(y-1)
    nodes[y-1].append(x-1)

depths=[-1]*N
# nxt[k][v] := 頂点vから2^k遡った祖先
nxt=list2d(17, N, -1)
# dfsで各頂点の深さと親を取得
def dfs(u, prev, depth):
    depths[u]=depth
    nxt[0][u]=prev
    for v in nodes[u]:
        if prev!=v:
            dfs(v, u, depth+1)

dfs(0, -1, 0)

# ダブリングのテーブル構築
for k in range(1, 17):
    for v in range(N):
        if nxt[k-1][v]==-1: 
            continue
        nxt[k][v]=nxt[k-1][nxt[k-1][v]]

# ab間の距離を取得
def get_dist(a, b):
    # aとbのLCAを取得
    lca=get_lca(a, b)
    # aの深さ+bの深さ-LCAの深さ*2でab間の距離
    return depths[a]+depths[b]-depths[lca]*2

def get_lca(a, b):
    # 深い方をbにする
    if depths[a]>depths[b]:
        a,b=b,a
    # bをaと同じ高さまで持ってくる
    gap=depths[b]-depths[a]
    for i in range(17):
        # ビットの立っている所を辿れば、好きな高さに移動できる
        if gap>>i&1:
            b=nxt[i][b]
    # この時点で一致すればそこがLCA
    if a==b:
        return a
    else:
        # aとbが一致する直前の高さまで持ってくる
        for i in range(16, -1, -1):
            a2=nxt[i][a]
            b2=nxt[i][b]
            if a2!=b2:
                a=a2
                b=b2
        # その親がLCA
        return nxt[0][a]

Q=INT()
for i in range(Q):
    a,b=MAP()
    # ab間の距離+1
    print(get_dist(a-1, b-1)+1)
