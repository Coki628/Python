"""
・自力AC！
・優先度付きキューとset、セグ木、重実装PAST系
・優先度付きキューとsetを組み合わせて、要素の追加・削除・最大値取得を動的に行う。
・上記構造で各幼稚園を管理して、各幼稚園最強をセグ木で管理する。
・最初pypyでもTLEして、setのremoveがpypy2のが速いっぽいって話を思い出して、
　pypy2で投げたら無事AC2.9秒。(制約3.5秒)
・高速化としてheapqの要素をタプルでない値にした。これでpypy3AC3.3秒、pypy2AC2.8秒。
"""

import sys
from heapq import heappush, heappop

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

N, Q = MAP()
M = 2 * 10**5

que = [[] for i in range(M)]
S = [set() for i in range(M)]
A = [0] * N
B = [0] * N
for i in range(N):
    a, b = MAP()
    b -= 1
    # 各園児の最初の所属とレート
    heappush(que[b], -(a*N+i))
    S[b].add(i)
    A[i] = a
    B[i] = b

seg = SegTree(M, min, INF)
for i in range(M):
    if que[i]:
        # 各幼稚園から最初の最大値を取得
        seg.update(i, -que[i][0]//N)

for _ in range(Q):
    c, d = MAP()
    c -= 1; d -= 1
    prevd = B[c]
    # 園児cの所属を更新
    S[B[c]].remove(c)
    S[d].add(c)
    B[c] = d
    heappush(que[d], -(A[c]*N+c))
    # heapqのremoveは同時にできないので、setの情報と照らし合わせて、取り出し時にやる
    while que[prevd] and -que[prevd][0]%N not in S[prevd]:
        heappop(que[prevd])
    # 変更のあった幼稚園prevdとdについてセグ木の値も更新する
    if que[prevd]:
        seg.update(prevd, -que[prevd][0]//N)
    else:
        seg.update(prevd, INF)
    seg.update(d, -que[d][0]//N)
    # 現在の全幼稚園中の最小値を取得する
    print(seg.all())
