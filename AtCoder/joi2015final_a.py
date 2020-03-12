# -*- coding: utf-8 -*-

"""
・pypyTLE,C++でAC0.2秒。
・区間最大、最小を取りながらDPする。
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

class SparseTable:

    def __init__(self, A, func):
        self.N = len(A)
        self.func = func
        h = 0
        while 1<<h < self.N:
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

        if l >= r:
            raise Exception
        a = self.height[r-l]
        return self.func(self.dat[a][l], self.dat[a][r-(1<<a)])

N, M, K = MAP()
A = LIST(N)

# 任意の区間での最大、最小をO(1)で出せるようにしておく
stmx = SparseTable(A, max)
stmn = SparseTable(A, min)
# dp[i] := オレンジをi個目まで詰めた時の最小コスト
dp = [INF] * (N+1)
dp[0] = 0
for i in range(N):
    for j in range(i+1, N+1):
        s = j - i
        if s > M:
            break
        mx = stmx.get(i, j)
        mn = stmn.get(i, j)
        # i個済の状態からs個詰める遷移
        dp[j] = min(dp[j], dp[i] + (K + s * (mx - mn)))
ans = dp[N]
print(ans)
