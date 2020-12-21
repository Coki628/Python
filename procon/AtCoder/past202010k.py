"""
・自力AC！
・BITと転倒数
・題意の数列を陽に持つと、クエリ内容によってはでかくなり過ぎる。
・ちょっと前にこどふぉコースで転倒数の数列マージするやつみたいのやってたのが良かった。(cf_edu_seg1_4c)
　ほぼその時と同じ感じのマージ関数書いた。
"""

import sys
from collections import Counter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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
MOD = 10**9
EPS = 10**-10

class BIT:
    """ Binary Indexed Tree """

    def __init__(self, n):
        self.n = n
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

    def query(self, l, r):
        """ 区間和の取得 [l, r) """
        return self.sum(r-1) - self.sum(l-1)
    
    def get(self, i):
        """ 一点取得 """
        return self.query(i, i+1)

    def update(self, i, x):
        """ 値の更新：添字i, 値x """
        self.add(i, x - self.get(i))

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

def get_inv(X):
    N = len(X)
    bit = BIT(22)
    res = 0
    for i in range(N):
        res += bit.query(X[i]+1, 22)
        res %= MOD
        bit.add(X[i], 1)
    return res

K = INT()
li = [[]] * K
CR = [None] * K
INV = [0] * K
for i in range(K):
    INT()
    li[i] = LIST()
    INV[i] = get_inv(li[i])
    CR[i] = Counter(li[i])

Q = INT()
B = [b-1 for b in LIST()]

def merge(C1, C2):
    C = [0] * 22
    acc = [0] * 22
    for i in range(1, 22):
        # それぞれの個数をマージ
        C[i] = C1[i] + C2[i]
        # 右側の累積和
        acc[i] = acc[i-1] + C2[i]
    # 今回のマージで増える転倒数
    inv = 0
    for i in range(2, 22):
        inv += acc[i-1] * C1[i]
        inv %= MOD
    return C, inv

X = []
CL = Counter()
ans = 0
for i in range(Q):
    ans += INV[B[i]]
    CL, inv = merge(CL, CR[B[i]])
    ans += inv
    ans %= MOD
print(ans)
