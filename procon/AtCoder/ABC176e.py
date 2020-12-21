"""
・自力AC！
・平面走査、セグ木
・基本的には平面走査みたいな感じで、行を1つずつ下げながら、
　各行での列の状態をまとめて見る、ってやる。
・計算量は30万にセグ木のlog、pypyAC1.29秒。(制約3秒なのでそんなに厳しくはない)
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
MOD = 10 ** 19 + 7
EPS = 10 ** -10

class SegTree:
    """
    セグメント木
    1.update: i番目の値をxに更新する
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
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.intv] * (n2 << 1)
        if A:
            for i in range(n):
                self.tree[n2+i] = A[i]
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

H, W, M = MAP()
# 各列に何個あるかを保持するセグ木
seg = SegTree(W+1, max, -INF, [0]*(W+1))
adjli = [[] for i in range(H+1)]
for i in range(M):
    h, w = MAP()
    seg.add(w, 1)
    adjli[h].append(w)

ans = 0
for h in range(1, H+1):
    rowcnt = len(adjli[h])
    # 今回の行の分を各列合計から除く
    for w in adjli[h]:
        seg.add(w, -1)
    ans = max(ans, rowcnt+seg.all())
    # 除いた分を戻す
    for w in adjli[h]:
        seg.add(w, 1)
print(ans)
