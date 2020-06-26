"""
・ここまで自力AC
・偶奇、累積和、区間和の最大値
・操作はリバースって言ってるけど、最終的に偶数位置を取ることを考えると、
　どこか1回だけある区間の取る要素を逆にできるので、それの最大値、になる。
・操作を奇数位置から始めるか偶数位置から始めるかで逆になる要素が変わるので両方やる。
・逆にした時の差分で累積和を取って、区間和の最大値を求める。
　左端固定して右端は区間maxを取りに行く、をやった。セグ木でもよかったけどSparse Tableにしてみた。
・でも考えたら左端を前から累積min、右端を後ろから累積maxしておけば線形でもいけたっぽいね。
"""

import sys
from itertools import accumulate

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
INF = 10 ** 19
MOD = 10 ** 9 + 7

class SparseTable:

    def __init__(self, A, func):
        self.N = len(A)
        self.func = func
        h = 0
        while 1<<h <= self.N:
            h += 1
        self.dat = list2d(h, 1<<h, 0)
        self.height = [0] * (self.N+1)

        for i in range(2, self.N+1):
            self.height[i] = self.height[i>>1] + 1
        for i in range(self.N):
            self.dat[0][i] = A[i]
        for i in range(1, h):
            for j in range(self.N):
                self.dat[i][j] = self.func(self.dat[i-1][j], self.dat[i-1][min(j+(1<<(i-1)), self.N-1)])
        
    def get(self, l, r):
        """ 区間[l,r)でのmin,maxを取得 """

        if l >= r or r > self.N:
            raise Exception
        a = self.height[r-l]
        return self.func(self.dat[a][l], self.dat[a][r-(1<<a)])

for _ in range(INT()):
    N = INT()
    A = LIST()

    B1 = []
    B2 = []
    for i in range(N-1):
        if i % 2 == 0:
            B1.append(A[i+1]-A[i])
        else:
            B2.append(A[i]-A[i+1])
    acc1 = [0] + list(accumulate(B1))
    acc2 = [0] + list(accumulate(B2))
    st1 = SparseTable(acc1, max)
    st2 = SparseTable(acc2, max)
    N1 = len(acc1)
    N2 = len(acc2)

    base = sum([a if i % 2 == 0 else 0 for i, a in enumerate(A)])
    ans = base
    for i in range(N1-1):
        l = acc1[i]
        r = st1.get(i+1, N1)
        ans = max(ans, base+r-l)
    for i in range(N2-1):
        l = acc2[i]
        r = st2.get(i+1, N2)
        ans = max(ans, base+r-l)
    print(ans)
