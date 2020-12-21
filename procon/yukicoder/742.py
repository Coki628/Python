"""
・自力AC！
・区間系、BIT
・BITとsetでうまくやる。
・まず左に行くやつ(以下L)と右に行くやつ(以下R)は、区間が重なっていれば必ずどこかで交わるので、
　区間の左端と右端を管理して、setとか使ってうまく数える。
・L同士、R同士は、動く速度が同じである以上、どちらかが止まってからでないと交わらない。
　これは、自分がどこまで行くかでそこまでに止まってくれてるやつの数をBITで数えたあと、
　自分もその先から来るやつに数えてもらえるように、BITの自分が止まる位置に値を足す。
・うまく区間管理して無事ACできたーよかったー、って思って想定解見たら、
　転倒数数えるだけでいけたらしい。。なんですと。。。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class BIT:
    """ Binary Indexed Tree """

    def __init__(self, n):
        # 0-indexed
        n += 1
        nv = 1
        while nv < n:
            nv *= 2
        self.n = n
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

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

N = INT()
L = [0] * (N+1)
L2 = [0] * (N+1)
R = [0] * (N+1)
R2 = [0] * (N+1)
for i in range(1, N+1):
    x = INT()
    if x <= i:
        L[i] = x
        L2[x] = i
    else:
        R[i] = x

ans = 0

# L - R
lcnt = set()
rcnt = set()
for i in range(1, N+1):
    if R[i]:
        rcnt.add(R[i])
        ans += len(lcnt)
    if L2[i]:
        lcnt.add(L2[i])
        ans += len(rcnt)
    if i in lcnt:
        lcnt.remove(i)
    if i in rcnt:
        rcnt.remove(i)

# L - L
bit = BIT(N+1)
for i in range(1, N+1):
    if L[i]:
        ans += bit.get(L[i], i+1)
        bit.add(L[i], 1)

# R - R
bit = BIT(N+1)
for i in range(N, 0, -1):
    if R[i]:
        ans += bit.get(i, R[i]+1)
        bit.add(R[i], 1)

print(ans)
