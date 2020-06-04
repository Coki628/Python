"""
・自力AC
・グラフ
・四角形になるサイクルを見つける。50頂点から4つ選ぶ組み合わせが23万くらいなので全探索できる。
・成立する条件は、4つそれぞれの頂点で他2つに距離1、残り1つに距離2、ならOK。
　と思ったけど、4つとも他2つに距離1なら残りが距離2は常に成立しているようなので、
　距離2の方はいらなかった。なのでワーシャルフロイドで全距離出したけど、
　それはいらなくて、繋がってる辺の情報だけあれば十分だった。
"""

import sys
from itertools import combinations

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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def warshall_floyd(graph: list) -> list:
    """ ワーシャルフロイド(頂点数, 隣接行列(0-indexed)) """
    from copy import deepcopy

    N = len(graph)
    res = deepcopy(graph)
    for i in range(N):
        res[i][i] = 0
    for k in range(N):
        for i in range(N):
            for j in range(N):
                res[i][j] = min(res[i][j], res[i][k] + res[k][j])
    for i in range(N):
        if res[i][i] < 0:
            return []
    return res

N, M = MAP()
G = list2d(N, N, INF)
for i in range(M):
    a, b = MAP()
    G[a][b] = 1
    G[b][a] = 1
wf = warshall_floyd(G)

ans = 0
for comb in combinations(range(N), 4):
    for i in range(4):
        C = [0] * 3
        for j in range(4):
            if i == j:
                continue
            a, b = comb[i], comb[j]
            if wf[a][b] <= 2:
                C[wf[a][b]] += 1
        if C[1] != 2 or C[2] != 1:
            break
    else:
        ans += 1
print(ans)
