# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/ttpc2019/editorial.pdf
　　　https://www.hamayanhamayan.com/entry/2019/09/01/114634_3
・DAG、経路DP
・2頂点が合流するまでのコストを合わせて、1つの新しい頂点とみなす。
・ゴール側は向き逆のグラフを使って同じことする。
・DAGはダイクストラしなくてもトポソ+DPでいける。
・今回は辺の向きが制約でs<tが保証されてるので、トポソも必要なくて若い頂点から順番にやればOK。
・やっぱりダイクストラより速くて、python1.7秒、pypy1.0秒AC。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = MAP()
w, x, y, z = MAP()
nodes = [[] for i in range(N+2)]
rev_nodes = [[] for i in range(N+2)]
for i in range(M):
    c, s, t = MAP()
    nodes[s].append((t, c))
    rev_nodes[t].append((s, c))

# w,yから各頂点
fromW = [INF] * (N+2)
fromW[w] = 0
for u in range(w, N+1):
    for v, c in nodes[u]:
        fromW[v] = min(fromW[v], fromW[u]+c)
fromY = [INF] * (N+2)
fromY[y] = 0
for u in range(y, N+1):
    for v, c in nodes[u]:
        fromY[v] = min(fromY[v], fromY[u]+c)

# 各頂点からx,z
toX = [INF] * (N+2)
toX[x] = 0
for u in range(x, 0, -1):
    for v, c in rev_nodes[u]:
        toX[v] = min(toX[v], toX[u]+c)
toZ = [INF] * (N+2)
toZ[z] = 0
for u in range(z, 0, -1):
    for v, c in rev_nodes[u]:
        toZ[v] = min(toZ[v], toZ[u]+c)

# 頂点wyから出る辺を先頭に追加
for i in range(1, N+1):
    nodes[0].append((i, fromW[i]+fromY[i]))
# 頂点xzへ向かうを末尾に追加
for i in range(1, N+1):
    nodes[i].append((N+1, toX[i]+toZ[i]))

fromWY = [INF] * (N+2)
fromWY[0] = 0
for u in range(N+2):
    for v, c in nodes[u]:
        fromWY[v] = min(fromWY[v], fromWY[u]+c)

# 共有部分がある時とない時で総コストの小さい方
ans = min(fromWY[N+1], fromW[x]+fromY[z])
if ans != INF:
    print(ans)
else:
    print('Impossible')
