# -*- coding: utf-8 -*-

"""
・最小全域木、Union-Find
・時間かかったけど自力AC！
・pythonで1.4秒くらいで、pypyはなぜか1.7秒くらいかかった。
・後からif grp == 1: break の条件つけたら0.05秒くらい改善。
・1回目のソートにキー指定したらこれも0.05秒くらい改善。
　合わせて0.1秒くらい速くなって1.3秒くらい。
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)

from operator import itemgetter

# Union-Find木
class UnionFind:
    def __init__(self, n):
        # 親要素のノード番号を格納。par[x] == xの時そのノードは根
        # 1-indexedのままでOK、その場合は[0]は未使用
        self.par = [i for i in range(n+1)]
        # 木の高さを格納する（初期状態では0）
        self.rank = [0] * (n+1)

    # 検索
    def find(self, x):
        # 根ならその番号を返す
        if self.par[x] == x:
            return x
        else:
            # 走査していく過程で親を書き換える
            self.par[x] = self.find(self.par[x])
            return self.par[x]

    # 併合
    def union(self, x, y):
        # 根を探す
        x = self.find(x)
        y = self.find(y)
        # 木の高さを比較し、低いほうから高いほうに辺を張る
        if self.rank[x] < self.rank[y]:
            self.par[x] = y
        else:
            self.par[y] = x
            # 木の高さが同じなら片方を1増やす
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    # 同じ集合に属するか判定
    def same(self, x, y):
        return self.find(x) == self.find(y)

N = int(input())
X = [[0] * 2 for i in range(N)]
Y = [[0] * 2 for i in range(N)]
for i in range(N):
    x, y = map(int, input().split())
    X[i][0] = x
    X[i][1] = i
    Y[i][0] = y
    Y[i][1] = i
# 元indexを保持して座標位置順にソート
X.sort(key=itemgetter(0))
Y.sort(key=itemgetter(0))

edges = []
for i in range(1, N):
    # 座標上で隣り合う頂点同士で辺集合を作る
    edges.append((X[i-1][1], X[i][1], X[i][0] - X[i-1][0]))
    edges.append((Y[i-1][1], Y[i][1], Y[i][0] - Y[i-1][0]))
# コスト順にソート
edges.sort(key=itemgetter(2))

uf = UnionFind(N)
ans = 0
grp = N
# Union-Findで閉路を作らない確認をして最小全域木を作る
for x, y, w in edges:
    if not uf.same(x, y):
        uf.union(x, y)
        ans += w
        grp -= 1
    # グループが1つになれば、それ以上やる必要ない
    if grp == 1:
        break
print(ans)
