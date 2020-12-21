"""
参考：みんなのツイート
・自力ならず。。なんか方針さっぱり出なかった。。
・二分探索応用、シミュレーション、順列数え上げ
・条件を満たす数列を作るには、xより大きい/小さい必要がある場所、の数をチェックする必要がある。
・これを求めるには、実際にその二分探索の操作をしたとして、答えの位置をposに合わせるためには、
　左/右に行く回数が何回あるか数えるといい。(この時、pos位置ちょうどはxに固定するので除く)
・この回数が分かれば、実際にxより大きい/小さい数がいくつあるかと照らし合わせて、
　nPr(実際にxより大きい/小さい数の個数, xより大きい/小さい数にしたい個数)をやって、
　残りはなんでもいいので階乗すればいい。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class CombTools:

    def __init__(self, MAX, MOD):

        MAX += 1
        self.MAX = MAX
        self.MOD = MOD
        factorial = [1] * MAX
        factorial[0] = factorial[1] = 1
        for i in range(2, MAX):
            factorial[i] = factorial[i-1] * i % MOD
        inverse = [1] * MAX
        inverse[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
        for i in range(MAX-2, -1, -1):
            inverse[i] = inverse[i+1] * (i+1) % MOD
        self.fact = factorial
        self.inv = inverse

    def nCr(self, n, r):

        if n < r: return 0
        r = min(r, n-r)
        numerator = self.fact[n]
        denominator = self.inv[r] * self.inv[n-r] % self.MOD
        return numerator * denominator % self.MOD

    def nPr(self, n, r):

        if n < r: return 0
        return self.fact[n] * self.inv[n-r] % self.MOD

# def calc(A, x):
#     l = 0
#     r = len(A)
#     while l < r:
#         m = (l+r) // 2
#         if A[m] <= x:
#             l = m + 1
#         else:
#             r = m
#     if l > 0 and A[l-1] == x:
#         return True
#     else:
#         return False

def calc(N, pos):
    l = 0
    r = N
    lcnt = rcnt = 0
    while l < r:
        m = (l+r) // 2
        if m <= pos:
            if m != pos:
                lcnt += 1
            l = m + 1
        else:
            if m != pos:
                rcnt += 1
            r = m
    return lcnt, rcnt

N, x, pos = MAP()

lneed, rneed = calc(N, pos)
ltotal = x - 1
rtotal = N - x

lrest = ltotal - lneed
rrest = rtotal - rneed
if lrest < 0 or rrest < 0:
    print(0)
    exit()

ct = CombTools(N, MOD)
others = lrest + rrest
ans = ct.nPr(ltotal, lneed) * ct.nPr(rtotal, rneed) * ct.fact[others] % MOD
print(ans)
