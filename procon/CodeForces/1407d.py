"""
参考：https://jupiro.hatenablog.com/entry/2020/09/09/021411
・自力ならず。。
・配る遷移と貰う遷移を両方やるDP、セグ木上の二分探索(Sparse Table使用)
・割と近いことは考えていたが、貰う方の遷移は考慮できなかった。
・セグ木の変わりにSparse Table使ったのでlog1つ落ちているはずだが、
　最大30万のNでpypyTLE。。2秒間に合わないのは遅いなー。。。
"""

import sys
from operator import le, ge

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
INF = 10**18
MOD = 10**9 + 7

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

    def bisearch_fore(self, l, r, x, func):
        """ 区間[l,r]で左から最初にxに対して比較の条件を満たすような値が出現する位置 """

        ok = r + 1
        ng = l - 1
        while ng+1 < ok:
            mid = (ok+ng) // 2
            if func(self.get(l, mid+1), x):
                ok = mid
            else:
                ng = mid
        if ok != r + 1:
            return ok
        else:
            return INF

    def bisearch_back(self, l, r, x, func):
        """ 区間[l,r]で右から最初にxに対して比較の条件を満たすような値が出現する位置 """

        ok = l - 1
        ng = r + 1
        while ok+1 < ng:
            mid = (ok+ng) // 2
            if func(self.get(mid, r+1), x):
                ok = mid
            else:
                ng = mid
        if ok != l - 1:
            return ok
        else:
            return -INF

N = INT()
A = LIST()

stmn = SparseTable(A, min)
stmx = SparseTable(A, max)

dp = [INF] * N
dp[0] = 0
for cur in range(N):
    if cur != 0:
        prev = stmn.bisearch_back(0, cur-1, A[cur], le)
        if prev != -INF:
            dp[cur] = min(dp[cur], dp[prev] + 1)
        prev = stmx.bisearch_back(0, cur-1, A[cur], ge)
        if prev != -INF:
            dp[cur] = min(dp[cur], dp[prev] + 1)
    if cur != N-1:
        nxt = stmn.bisearch_fore(cur+1, N-1, A[cur], le)
        if nxt != INF:
            dp[nxt] = min(dp[nxt], dp[cur] + 1)
        nxt = stmx.bisearch_fore(cur+1, N-1, A[cur], ge)
        if nxt != INF:
            dp[nxt] = min(dp[nxt], dp[cur] + 1)
ans = dp[-1]
print(ans)
