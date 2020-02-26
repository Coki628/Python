# -*- coding: utf-8 -*-

"""
参考：https://qnighy.github.io/informatics-olympiad/joi2011-day4-bookshelf-comment.html
　　　http://algoogle.hadrori.jp/joi/sc2011bookshelf/
　　　http://h0rnet.hatenablog.com/entry/2016/03/14/JOI_11_Day4_Bookshelf_(_DP_%2B_Segment_Tree_)
・蟻本演習3-3-1
・セグメント木
・動かすコストの最小値 = 全コスト - 動かさないコストの最大値
　から、答えを求める。でも言い換え考察力要るわー。
・pythonTLE,pypyMLE,メモリ制限64Mつらい。
・少しは軽くなるかと思ってセグ木の代わりにBIT使ったけど変化なし。。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

class BIT:
    """ BIT汎用版 """

    def __init__(self, n, func, init):
        # 0-indexed
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.func = func
        self.init = init
        self.tree = [init] * nv
    
    def query(self, i):
        """ [0, i]の値を取得 """
        s = self.init
        i += 1
        while i > 0:
            s = self.func(s, self.tree[i-1])
            i -= i & -i
        return s
    
    def update(self, i, x):
        """ 値の更新：添字i, 値x """
        i += 1
        while i <= self.size:
            self.tree[i-1] = self.func(self.tree[i-1], x)
            i += i & -i

N = INT()
cost = LIST(N)
A = [a-1 for a in LIST(N)]

# bit[A[i]] := 前からiまで見て、A[i]を動かさない時の、動かさない総コストの最大値
bit = BIT(N, max, 0)
total = sum(cost)
mx = 0
for i in range(N):
    a = A[i]
    res = bit.query(a-1)
    bit.update(a, res+cost[a])
mx = bit.query(N-1)
# 全て動かす場合 - 動かさない総コストの最大 = 動かすコストの最小
print((total-mx)*2)
