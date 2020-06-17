"""
・ちょっとツイートで見かけたから久々に解いた。自力ACできてよかった。
・グラフに帰着、DAG、メモ化再帰、閉路検出、最長パス
・各試合を頂点とみなす。今回はこの始点が見えた。
・閉路検出は、確定前の頂点にもう1回来ちゃったらNG、でやった。
・N*Nの半分で50万のメモ化再帰はTLEが心配だったけどなんとかpypyAC1.7秒。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

def dfs(u):
    # 閉路があったら即NG
    if memo[u] == -2:
        print(-1)
        exit()
    if memo[u] != -1:
        return memo[u]
    memo[u] = -2
    res = 0
    for v in nodes[u]:
        # max取って最長パスにする
        res = max(res, dfs(v))
    memo[u] = res + 1
    return res + 1

N = INT()
NN = N * N
# 2重辺にならないようにsetで作る
nodes = [set() for i in range(NN+1)]
for a in range(N):
    B = [b-1 for b in LIST()]
    for j in range(N-2):
        a1 = a2 = a
        b1 = B[j]
        b2 = B[j+1]
        if a1 > b1:
            a1, b1 = b1, a1
        if a2 > b2:
            a2, b2 = b2, a2
        # 次の試合への辺を張る
        nodes[a1*N+b1].add(a2*N+b2)
        # 始点があると楽なので最初の試合は始点から繋ぐ
        if j == 0:
            nodes[NN].add(a1*N+b1)

memo = [-1] * (NN+1)
# 始点の分を引けば答え
ans = dfs(NN) - 1
print(ans)
