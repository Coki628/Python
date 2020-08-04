"""
・タプルのソートがおそい件について
・バケットソートしてみたらめっちゃ速くなった！
・pypyAC1.99秒→1.19秒。
"""

import sys
from operator import itemgetter

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
EPS = 10 ** -10

class BIT:
    """ Binary Indexed Tree """

    def __init__(self, n):
        # 0-indexed
        n += 1
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.tree = [0] * nv

    def sum(self, i):
        """ [0, i]を合計する """
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i-1]
            i -= i & -i
        return s

    def add(self, i, x):
        """ 値の追加：添字i, 値x """
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            i += i & -i

    def get(self, l, r=None):
        """ 区間和の取得 [l, r) """
        # 引数が1つなら一点の値を取得
        if r is None: r = l + 1
        res = 0
        if r: res += self.sum(r-1)
        if l: res -= self.sum(l-1)
        return res

    def update(self, i, x):
        """ 値の更新：添字i, 値x """
        self.add(i, x - self.get(i))

def bucket_sort(A, M, key):
    from itertools import accumulate

    N = len(A)
    C = [0] * (M+1)
    res = [()] * N
    for i in range(N):
        C[A[i][key]] += 1
    C = list(accumulate(C))

    for i in range(N-1, -1, -1):
        res[C[A[i][key]]-1] = A[i]
        C[A[i][key]] -= 1
    return res

N, Q = MAP()
A = [a-1 for a in LIST()]

prev = [-1] * N
events = []
bit = BIT(N)
for t, a in enumerate(A):
    s = prev[a]
    if s != -1:
        events.append((t, 0, s))
    prev[a] = t

for i in range(Q):
    l, r = MAP()
    l -= 1; r -= 1
    events.append((r, 1, l, i))

# events.sort(key=itemgetter(0))
events = bucket_sort(events, N, 0)
ans = [0] * Q
for event in events:
    # 色の区間が来たら、点を追加
    if event[1] == 0:
        t, _, s = event
        bit.add(s, 1)
    # クエリが来たら、点の数を数える
    else:
        r, _, l, i = event
        total = r - l + 1
        # 区間全体 - この時の領域内の点の数
        ans[i] = total - bit.get(l, N)

for a in ans:
    print(a)
