# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/70779
・いもす応用、index取得セグ木
・区間の終端が遅い方を優先して消すと、貪欲できる典型
・区間の管理で、区間数だけでなくどの区間が今残ってるかのindexも保持する方法として
　今回のやり方はすごくためになった。
・20万logでpypyAC1.6秒。
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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

class SegTreeIndex:
    """
    セグメント木(index取得対応版)
    1.update:  i番目の値をxに更新する
    2.query: 区間[l, r)の値とindex(同値があった場合は一番左)を得る
    """
 
    def __init__(self, n, func, init):
        """
        :param n: 要素数(0-indexed)
        :param func: 値の操作に使う関数(min, max)
        :param init: 要素の初期値(単位元)
        """
        self.n = n
        self.func = func
        self.init = init
        # nより大きい2の冪数
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.init] * (n2 << 1)
        self.index = [self.init] * (n2 << 1)
        # 1段目(最下段)の初期化
        for i in range(n2):
            self.index[i+n2] = i
        # 2段目以降の初期化
        for i in range(n2-1, -1, -1):
            # 全部左の子の値に更新
            self.index[i] = self.index[i*2]

    def update(self, i, x):
        """
        i番目の値をxに更新
        :param i: index(0-indexed)
        :param x: update value
        """
        i += self.n2
        self.tree[i] = x
        while i > 1:
            left, right = min(i, i^1), max(i, i^1)
            if self.func(self.tree[left], self.tree[right]) == self.tree[left]:
                self.tree[i >> 1] = self.tree[left]
                self.index[i >> 1] = self.index[left]
            else:
                self.tree[i >> 1] = self.tree[right]
                self.index[i >> 1] = self.index[right]
            i >>= 1
 
    def query(self, a, b):
        """
        [a, b)の値を得る
        :param a: index(0-indexed)
        :param b: index(0-indexed)
        """
        l = a + self.n2
        r = b + self.n2
        s = (self.init, -1)
        while l < r:
            if r & 1:
                r -= 1
                res = self.func(s[0], self.tree[r])
                # 左との一致を優先する
                if res == s[0]:
                    pass
                else:
                    s = (self.tree[r], self.index[r])
            if l & 1:
                res = self.func(self.tree[l], s[0])
                # 左との一致を優先する
                if res == self.tree[l]:
                    s = (self.tree[l], self.index[l])
                else:
                    pass
                l += 1
            l >>= 1
            r >>= 1
        return s

N, K = MAP()
MAX = 200007

LRs = [[] for i in range(MAX+2)]
R = [0] * (N+1)
for i in range(N):
    l, r = MAP()
    # いもすっぽい始点終点の追加を、隣接リストっぽくやってindexを保持する
    LRs[l].append(i+1)
    # 終点位置は負数で保持することで、始点と区別できる
    LRs[r+1].append(-(i+1))
    R[i+1] = r

# index毎に、終点位置を保持するセグ木
sti = SegTreeIndex(N+1, max, -INF)
segcnt = 0
ans = []
removed = set()
for i in range(1, MAX+2):
    # この地点での始点と終点の操作
    for idx in LRs[i]:
        # 始点
        if idx > 0:
            sti.update(idx, R[idx])
            segcnt += 1
        # 終点
        else:
            idx = abs(idx)
            # 既に削除されていないかチェック
            if idx not in removed:
                sti.update(idx, -INF)
                segcnt -= 1
    # 区間数がKを超えている場合、削除をやる
    while segcnt > K:
        # 現在残っている区間の中で、最も終点位置が大きいもののindex
        _, idx = sti.query(0, N+1)
        sti.update(idx, -INF)
        ans.append(idx)
        removed.add(idx)
        segcnt -= 1
print(len(ans))
print(*ans)
