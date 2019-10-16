# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2018/09/15/193740
　　　https://camypaper.bitbucket.io/2016/12/18/adc2016/
　　　https://topcoder.g.hatena.ne.jp/iwiwi/20131226/1388062106
・蟻本演習2-4-3
・部分永続Union-Find、データ構造をマージする一般的なテク
・しかしTLE。。
・これは経路圧縮しないlogNのUnion-Findがベースなので、
　再帰が弱いpythonだとあんまいけてないのかも。。
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

class PartiallyPersistentUnionFind:
    """ 部分永続Union-Find木 """

    def __init__(self, n):
        self.n = n
        # xが根のときはxを含むグループのサイズ(の-1倍)、そうでないときは親ノード
        self.par = [-1] * (n+1)
        # 最後に「根」ではなくなった瞬間の時刻
        self.last = [-1] * (n+1)

    def find(self, t, x):
        """ 根の検索(グループ番号と言えなくもない) """
        # 根ならその番号を返す
        if self.last[x] == -1 or t < self.last[x]:
            return x
        else:
            return self.find(t, self.par[x])

    def union(self, t, x, y):
        """ 併合 """
        # 根を探す
        x = self.find(t, x)
        y = self.find(t, y)
        if x == y:
            return False
        # 要素数の少ない方を付け替える(マージテク)
        if self.par[x] > self.par[y]:
            x, y = y, x
        # xにyを付ける
        self.par[y] = x
        self.last[y] = t
        self.par[x] += self.par[y]
        return True

    def same(self, t, x, y):
        """ 同じ集合に属するか判定 """
        return self.find(t, x) == self.find(t, y)

def bisearch_min(mn, mx, func):
    """ 条件を満たす最小値を見つける二分探索 """

    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            # 下を探しに行く
            ok = mid
        else:
            # 上を探しに行く
            ng = mid
    return ok

def calc(m):
    # m回併合して、xとyが同じ集合に属しているかどうか
    return ppuf.same(m, x, y)

N, M = MAP()
ppuf = PartiallyPersistentUnionFind(N)
for i in range(M):
    a, b = MAP()
    if not ppuf.same(i, a, b):
        ppuf.union(i+1, a, b)

for _ in range(INT()):
    x, y = MAP()
    if not ppuf.same(M+1, x, y):
        print(-1)
    else:
        res = bisearch_min(-1, M+1, calc)
        if res > M:
            print(-1)
        else:
            print(res)
