# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/kyort0n/status/1160206692396593152
　　　https://twitter.com/segment_turai/status/116020994997407744
・ベルマンフォード
・これ解いてる人少なかったし、本番中見なかったんだけど見とけば良かったー。
　終了後、ベルマンフォード使うっていうだけネタバレしてたんだけど、
　あと全部自力で(ABC061dを参考)割とさくっと解けたし、自力ACいけたんじゃないかなぁ。。
　→あとから解説でも修正出てたけど、厳密にはこの負閉路チェックだと落ちるケースがある。
　→上記件をtwitterの情報を参考に修正した。問題のケースも通るし、ACもするけど、正しいかはそんなに自信ない。
　→コンテスト後追加されたテストケースでもACを確認。これなら多分大丈夫だろう。
・計算量はNMで2500*5000=1250万、pypyで0.6秒AC。
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

def bellman_ford(N: int, edges: list, src: int) -> list:
    """ ベルマンフォード(頂点数, 辺集合(0-indexed), 始点) """

    # 頂点[ある始点からの最短距離] (経路自体を知りたい時はここに前の頂点も持たせる)
    res = [INF] * N
    res[src] = 0
    # 各辺によるコストの置き換えを頂点数N-1回繰り返す
    for i in range(N-1):
        for src, dest, cost in edges:
            if res[dest] > res[src] + cost:
                res[dest] = res[src] + cost
    # 頂点Nが負の閉路に含まれていないかチェックする
    for i in range(N):
        for src, dest, cost in edges:
            if res[dest] > res[src] + cost:
                # 検出した負閉路はすぐに-INFにする
                res[dest] = -INF
                if dest == N-1:
                    # 該当していたら空リストを返却
                    return []
    # 問題なければ頂点リストを返却
    return res

N, M, P = MAP()
edges = []
for i in range(M):
    a, b, c = MAP()
    # Pは最後に請求される、だけど、ようは動く度に支払ってると思ってもいい
    edges.append((a-1, b-1, -c+P))

res = bellman_ford(N, edges, 0)
if not len(res):
    print(-1)
else:
    # 正負を直す
    ans = -res[N-1]
    # スコアがマイナスなら0
    print(max(0, ans))
