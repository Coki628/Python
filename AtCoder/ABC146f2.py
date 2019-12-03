# -*- coding: utf-8 -*-

"""
参考：https://www.youtube.com/watch?v=7IlBVSglZqc
・一応やってみた、貪欲じゃない解。
・辞書順最小系、後ろからDP、セグ木で遷移
・計算量10万log、pythonAC1.6秒、pypyAC0.4秒。やっぱセグ木のlogはpypyのが全然速いね。
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

class SegTree:
    """
    セグメント木
    1.update:  i番目の値をxに更新する
    2.query: 区間[l, r)の値を得る
    """

    def __init__(self, n, func, intv, A=[]):
        """
        :param n: 要素数(0-indexed)
        :param func: 値の操作に使う関数(min, max, add, gcdなど)
        :param intv: 要素の初期値(単位元)
        :param A: 初期化に使うリスト(オプション)
        """
        self.n = n
        self.func = func
        self.intv = intv
        # nより大きい2の冪数
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.intv] * (n2 << 1)
        # 初期化の値が決まっている場合
        if A:
            # 1段目(最下段)の初期化
            for i in range(n):
                self.tree[n2+i] = A[i]
            # 2段目以降の初期化
            for i in range(n2-1, -1, -1):
                self.tree[i] = self.func(self.tree[i*2], self.tree[i*2+1])

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
            self.tree[i] = self.func(self.tree[i*2], self.tree[i*2+1])

    def query(self, a, b):
        """
        [a, b)の値を得る
        :param a: index(0-indexed)
        :param b: index(0-indexed)
        """
        l = a + self.n2
        r = b + self.n2
        s = self.intv
        while l < r:
            if r & 1:
                r -= 1
                s = self.func(s, self.tree[r])
            if l & 1:
                s = self.func(s, self.tree[l])
                l += 1
            l >>= 1
            r >>= 1
        return s

    def get(self, i):
        """ 一点取得 """
        return self.tree[i+self.n2]

    def all(self):
        """ 全区間[0, n)の取得 """
        return self.tree[1]

N, M = MAP()
S = input()

# dp[i] := iマスからNマスへの最短手数
dp = SegTree(N+1, min, INF)
dp.update(N, 0)
for i in range(N-1, -1, -1):
    # ゲームオーバーマスには遷移できない
    if S[i] == '1':
        continue
    mn = dp.query(i+1, min(N+1, i+M+1))
    if mn != INF:
        dp.update(i, mn+1)

# 辿り着けない
if dp.get(0) == INF:
    print(-1)
    exit()

cnt = dp.get(0)
cur = 0
prev = -1
ans = []
for i in range(1, N+1):
    if dp.get(i) == INF:
        continue
    # 手数が変わる場所でなるべく手前に飛ぶ
    if dp.get(i) != cnt:
        prev = cur
        cur = i
        ans.append(cur-prev)
        cnt = dp.get(i)
print(*ans)
