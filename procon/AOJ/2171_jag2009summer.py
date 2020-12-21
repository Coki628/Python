"""
参考：https://drken1215.hatenablog.com/entry/2019/03/20/000200
・蟻本演習4-1-1
・ランダムウォーク、連立一次方程式、ガウス・ジョルダン法、期待値、最短経路応用
・色々やる。大変。
・まず次に向かう頂点として、標識がある交差点の場合は最短経路に含まれるルートでないといけない。
　ダイクストラの最短経路は普通、本当は複数あっても、適当な最短経路を1個出すだけなので、
　まずここからして、どうすればええんじゃあ、となる。ルートを逆に辿ると復元はできるけど、
　これも結局ある1つの最短経路なので、情報として足りない。
・で、方法としては、終点からダイクストラ、だった。これをやると、
　任意の頂点から終点に向かう最短経路が求まるので、ある頂点から次の頂点を目指す時、
　その辺を使って次の頂点に行った時に、ちゃんと最短ルートのコストになってればOK、
　そうでないならダメ、と判定できる。
・行列の構築は期待値DPみたいな式を作って、行列に当てはまるような形に式変形、ってなことをやる。
　今回は、E[u] := 頂点uから頂点tへのコストの期待値、とした時に、
　E[u] := sum(E[v] + G[u][v])/K (vはuから向かう各頂点、Kはその向かう頂点数)
　みたいになる。このまま期待値DPしたくなるような式だけど、このままだと行列にはまらない。
　右辺を定数にしたいなぁという気持ちになると、E[u]*K-sum(E[v]) = sum(G[u][v]) とできる。
　これなら行列上では自分のところに*K、向かう頂点に*-1、と当てはめられる。
　これで全部やると解なし。。。
　E[t]は終点で最初からコストの期待値は0と決まっているので、これを入れないといけない。
　なんかどうしたらいいか分からなくてけんちょんさんのソースコード見ちゃったけど、
　E[t] = 0、は左辺に1*E[t]、右辺に0でいいんだな。確かにそれはそうっぽいね。。
・ここまで全部整えたら無事AC。解説だと、自己ループに対してさらに式変形する説明がされていて、
　問題文見ても確かに自己ループない保証しないよって書いてあるんだけど、
　なんかテストケース覗いたらとりあえずなさそうだったし、ひとまず1回投げてみたら通ったんだよね。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def dijkstra(G, src):
    """ O(V^2)のダイクストラ """

    N = len(G)
    dist = [INF] * N
    used = [False] * N

    dist[src] = 0
    while 1:
        v = -1
        for u in range(N):
            if not used[u] and (v == -1 or dist[u] < dist[v]):
                v = u
        if v == -1:
            break
        used[v] = True
        for u in range(N):
            dist[u] = min(dist[u], dist[v] + G[v][u])
    return dist

def gauss_jordan(A, b):
    """ ガウス・ジョルダン法(連立方程式の解) """

    N = len(A)
    B = list2d(N, N+1, 0)
    for i in range(N):
        for j in range(N):
            B[i][j] = A[i][j]
    for i in range(N):
        B[i][N] = b[i]
    
    for i in range(N):
        pivot = i
        for j in range(i, N):
            if abs(B[j][i]) > abs(B[pivot][i]):
                pivot = j
        B[i], B[pivot] = B[pivot], B[i]

        if abs(B[i][i]) < EPS:
            return []

        for j in range(i+1, N+1):
            B[i][j] /= B[i][i]
        for j in range(N):
            if i != j:
                for k in range(i+1, N+1):
                    B[j][k] -= B[j][i] * B[i][k]
    
    res = [0] * N
    for i in range(N):
        res[i] = B[i][N]
    return res

while 1:
    N, s, t = MAP()
    if N == s == t == 0:
        break

    has_sign = LIST()
    s -= 1
    t -= 1
    G = list2d(N, N, INF)
    for i in range(N):
        for j, c in enumerate(LIST()):
            if c:
                G[i][j] = c

    # 終点からの最短経路にする
    dist = dijkstra(G, t)
    if dist[s] == INF:
        print('impossible')
        continue

    nxt = [[] for i in range(N)]
    for u in range(N):
        if u == t:
            continue
        if has_sign[u]:
            for v in range(N):
                # この辺G[u][v]がtへの最短経路とイコールなら追加
                if G[u][v] != INF and dist[u]-G[u][v] == dist[v]:
                    nxt[u].append(v)
        else:
            for v in range(N):
                if G[u][v] != INF:
                    nxt[u].append(v)

    # 行列の構築
    mat = list2d(N, N, 0)
    B = [0] * N
    for u in range(N):
        # 終点は0確定なので例外処理
        if u == t:
            mat[u][u] = 1
            B[u] = 0
        else:
            # 係数は式変形より、自分にk、向かう各頂点に-1、右辺は向かう頂点へのコスト総和
            k = len(nxt[u])
            mat[u][u] = k
            for v in nxt[u]:
                mat[u][v] = -1
                B[u] += G[u][v]

    res = gauss_jordan(mat, B)
    ans = res[s]
    print(ans)
