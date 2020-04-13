# -*- coding: utf-8 -*-

"""
・ちょっと時間かかったけどなんとか自力AC！
・区間で色々うまくやる系
・indexセグ木、久し振りに使った。
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
        while i > 0:
            i >>= 1
            left, right = i*2, i*2+1
            if self.func(self.tree[left], self.tree[right]) == self.tree[left]:
                self.tree[i] = self.tree[left]
                self.index[i] = self.index[left]
            else:
                self.tree[i] = self.tree[right]
                self.index[i] = self.index[right]
 
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

    def get(self, i):
        """ 一点取得 """
        return (self.tree[i+self.n2], self.index[i+self.n2])

    def all(self):
        """ 全区間[0, n)の取得 """
        return (self.tree[1], self.index[1])

N = INT()
A = [0] + LIST()

# まず経過日数より多い分は無意味なので削る
for i in range(N+1):
    A[i] = min(A[i], i)

sti = SegTreeIndex(N+1, max, 0)
for i in range(N):
    sti.update(i, A[i])
# 最終日は絶対行かなきゃいけないのでそれを初期値とする
mx = A[-1]
cur = N - mx
ans = 1
while cur > 0:
    # 現在日より後ろで一番多く減らせる日を取得
    mx, idx = sti.query(cur, N+1)
    cur -= mx
    ans += 1
    # 使ってしまった日は0にしておく
    sti.update(idx, 0)
print(ans)
