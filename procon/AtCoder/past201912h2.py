# -*- coding: utf-8 -*-

"""
・ちょっと工夫して区間加算はしなくていいようにした。
　普通のセグ木で区間最小は取得してるけど。
・さすがにこっちのがかなり速くて、pypyAC0.6秒、pythonでもAC1.6秒。
　1点取得とか全区間取得はO(1)で取れるようにしてるしね。
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

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

N = INT()
A = LIST()

N0 = ceil(N, 2)
N1 = N // 2
A0 = []
A1 = []
for i in range(N):
    if i % 2 == 0:
        A0.append(A[i])
    else:
        A1.append(A[i])
# 偶奇に分けて値を持っておく
st0 = SegTree(N0, min, INF, A0)
st1 = SegTree(N1, min, INF, A1)

Q = INT()
cnt = 0
# 全区間について減る値を持っておく
all0 = 0
all1 = 0
for _ in range(Q):
    q = LIST()
    if q[0] == 1:
        _, x, a = q
        x -= 1
        if x % 2 == 0:
            if st0.get(x//2) - all0 >= a:
                st0.update(x//2, st0.get(x//2) - a)
                cnt += a
        else:
            if st1.get(x//2) - all1 >= a:
                st1.update(x//2, st1.get(x//2) - a)
                cnt += a
    elif q[0] == 2:
        _, a = q
        # 奇数番目(indexだと偶数)の値を一括更新
        if st0.all() - all0 >= a:
            all0 += a
            cnt += a * N0
    else:
        _, a = q
        if N > 1:
            if st0.all() - all0 >= a and st1.all() - all1 >= a:
                all0 += a
                all1 += a
                cnt += a * N
        else:
            if st0.all() - all0 >= a:
                all0 += a
                cnt += a * N
print(cnt)
