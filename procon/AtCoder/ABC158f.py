# -*- coding: utf-8 -*-

"""
・終了後、ほぼ自力AC(ツイートから、キーワードだけ聞いてた。)
・後ろからDP、セグ木で前計算
・ヒントを得ていたとはいえ、ほぼ真っ直ぐ正しい方針で実装して通せた。
　コンテスト中こっち優先で考えてたら、時間内に通せてたかもと思うとちょっと残念。。
・計算量は20万にlog、案外きつくてpypyAC1.5秒。
"""

import sys
from bisect import bisect_left

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
MOD = 998244353

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
XD = []
for _ in range(N):
    x, d = MAP()
    XD.append((x, d))

XD.sort()
X, _ = zip(*XD)
X = list(X)
X += [INF]
# st[i] := ロボットiが動くことで、何番目のロボットまで連動して動くか
st = SegTree(N+1, max, -INF)
st.update(N, N)
for i in range(N-1, -1, -1):
    x, d = XD[i]
    j = bisect_left(X, x+d)
    mx = st.query(i, j)
    st.update(i, max(i, mx))

# dp[i] := i番目のロボットまで(後ろから)見て、そこから後ろの状態の通り数
dp = [0] * (N+1)
dp[N] = 1
for i in range(N-1, -1, -1):
    # i番目を動かさない遷移(1つ前の状態全てを使える)
    dp[i] += dp[i+1]
    # i番目を動かす遷移(自分が影響する範囲は自由に変えられないので、その1つ前から持ってくる)
    dp[i] += dp[st.get(i)+1]
    dp[i] %= MOD
ans = dp[0]
print(ans)
