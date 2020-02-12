# -*- coding: utf-8 -*-

"""
参考：https://qiita.com/drken/items/7f98315b56c95a6181a4
・蟻本演習3-5-5
・グリッドで2部グラフ、最大マッチング、最大安定集合
・グリッドの偶奇で2部グラフを構成する。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

class BipartiteMatching:
    """
    XとYの二部グラフの最大マッチング X={0,1,2,...|X|-1} Y={0,1,2,...,|Y|-1}
    edges[x]: xとつながるYの頂点のset
    match1[x]: xとマッチングされたYの頂点
    match2[y]: yとマッチングされたXの頂点
    """

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.edges = [set() for _ in range(n)]
        self.match1 = [-1] * n
        self.match2 = [-1] * m
 
    def dfs(self, v, visited):
        """
        :param v: X側の未マッチングの頂点の1つ
        :param visited: 空のsetを渡す（外部からの呼び出し時）
        :return: 増大路が見つかればTrue
        """
        for u in self.edges[v]:
            if u in visited:
                continue
            visited.add(u)
            if self.match2[u] == -1 or self.dfs(self.match2[u], visited):
                self.match2[u] = v
                self.match1[v] = u
                return True
        return False
    
    def add(self, a, b):
        self.edges[a].add(b)

    def whois1(self, a):
        """ :param: グループ1の頂点 :return: ペアになるグループ2の頂点 """
        return self.match1[a]

    def whois2(self, a):
        """ :param: グループ2の頂点 :return: ペアになるグループ1の頂点 """
        return self.match2[a]

    def solve(self):
        # 増大路発見に成功したらTrue(=1)。合計することでマッチング数となる
        return sum(self.dfs(i, set()) for i in range(self.n))

def build_grid(H, W, intv, _type, space=True, padding=False):
    # 入力がスペース区切りかどうか
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
    # 余白の有無
    if padding:
        offset = 1
    else:
        offset = 0
    grid = list2d(H+offset*2, W+offset*2, intv)
    for i in range(offset, H+offset):
        row = _list()
        for j in range(offset, W+offset):
            grid[i][j] = row[j-offset]
    return grid

H, W = MAP()
grid = build_grid(H, W, '*', str, space=0, padding=1)

H += 2; W += 2
HW = H * W
# 簡単のため、全頂点を両方の集合に作る(辺を繋ぐ時に2部になってれば問題ない)
bm = BipartiteMatching(HW, HW)
total = 0
for h in range(1, H-1):
    for w in range(1, W-1):
        if grid[h][w] == '.':
            total += 1
            if grid[h-1][w] == '.':
                # グリッドの偶奇で2部グラフを構成する
                if (h+w) % 2 == 0:
                    bm.add(h*W+w, (h-1)*W+w)
                else:
                    bm.add((h-1)*W+w, h*W+w)
            if grid[h][w-1] == '.':
                if (h+w) % 2 == 0:
                    bm.add(h*W+w, h*W+(w-1))
                else:
                    bm.add(h*W+(w-1), h*W+w)

# 最大安定集合 = 有効な全頂点数 - 最大マッチング
ans = total - bm.solve()
print(ans)
