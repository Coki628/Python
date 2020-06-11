"""
・結局客が探すのは1,2列目だけなので、そこだけセグ木で最大とindex取れるようにして管理。
・実装おつ。バグらせず通せて良かった。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

class SegTreeIndex:
    """
    セグメント木(index取得対応版)
    1.update:  i番目の値をxに更新する
    2.query: 区間[l, r)の値とindex(同値があった場合は一番左)を得る
    """

    def __init__(self, n, func, intv, A=[]):
        """
        :param n: 要素数(0-indexed)
        :param func: 値の操作に使う関数(min, max)
        :param intv: 要素の初期値(単位元)
        :param A: 初期化に使うリスト(オプション)
        """
        self.n = n
        self.func = func
        self.intv = (intv, -1)
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.intv] * (n2 << 1)
        if A:
            for i in range(n):
                self.tree[n2+i] = (A[i], i)
            for i in range(n2-1, -1, -1):
                self.tree[i] = self.compare(self.tree[i*2], self.tree[i*2+1])

    def compare(self, a, b):
        if a[0] == b[0]:
            if a[1] <= b[1]:
                return a
            else:
                return b
        elif self.func(a[0], b[0]) == a[0]:
            return a
        else:
            return b

    def update(self, i, x):
        """
        i番目の値をxに更新
        :param i: index(0-indexed)
        :param x: update value
        """
        i += self.n2
        self.tree[i] = (x, i-self.n2)
        while i > 0:
            i >>= 1
            self.tree[i] = self.compare(self.tree[i*2], self.tree[i*2+1])

    def add(self, i, x):
        self.update(i, self.get(i) + x)

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
                s = self.compare(s, self.tree[r])
            if l & 1:
                s = self.compare(s, self.tree[l])
                l += 1
            l >>= 1
            r >>= 1
        return s

    def get(self, i):
        """ 一点取得 """
        return self.tree[i+self.n2][0]

    def all(self):
        """ 全区間[0, n)の取得 """
        return self.tree[1]

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

N = INT()
sold = [0] * N
A = [[]] * N
row1 = [0] * N
row2 = [0] * N
for i in range(N):
    A[i] = LIST()[1:]
    row1[i] = A[i][0]
    if len(A[i]) >= 2:
        row2[i] = A[i][1]
M = INT()
B = LIST()

st1 = SegTreeIndex(N, max, -INF, row1)
st2 = SegTreeIndex(N, max, -INF, row2)
ans = [0] * M
for i, b in enumerate(B):
    if b == 1:
        val, idx = st1.all()
        ans[i] = val
        sold[idx] += 1
        nxt = sold[idx]
        if nxt < len(A[idx]):
            st1.update(idx, A[idx][nxt])
        else:
            st1.update(idx, 0)
        if nxt+1 < len(A[idx]):
            st2.update(idx, A[idx][nxt+1])
        else:
            st2.update(idx, 0)
    else:
        val1, idx1 = st1.all()
        val2, idx2 = st2.all()
        if val1 > val2:
            ans[i] = val1
            sold[idx1] += 1
            nxt = sold[idx1]
            if nxt < len(A[idx1]):
                st1.update(idx1, A[idx1][nxt])
            else:
                st1.update(idx1, 0)
            if nxt+1 < len(A[idx1]):
                st2.update(idx1, A[idx1][nxt+1])
            else:
                st2.update(idx1, 0)
        else:
            ans[i] = val2
            sold[idx2] += 1
            nxt = sold[idx2]
            if nxt+1 < len(A[idx2]):
                st2.update(idx2, A[idx2][nxt+1])
            else:
                st2.update(idx2, 0)
for a in ans:
    print(a)
