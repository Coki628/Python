"""
参考：https://koboshi-kyopro.hatenablog.com/entry/2020/06/28/015323
・自力ならず。。
・数え上げ、組み合わせ、順列、包除原理
・包除、気付かない。。考え方は写像の3相とだいたい同じ。
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
EPS = 10 ** -10

class ModTools:
    """ 階乗・逆元用のテーブルを構築する """

    def __init__(self, MAX, MOD):

        # nCrならn、nHrならn+rまで作る
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
        """ 組み合わせ """

        if n < r: return 0
        r = min(r, n-r)
        numerator = self.fact[n]
        denominator = self.inv[r] * self.inv[n-r] % self.MOD
        return numerator * denominator % self.MOD

    def nHr(self, n, r):
        """ 重複組み合わせ """

        return self.nCr(r+n-1, r)

    def nPr(self, n, r):
        """ 順列 """

        if n < r: return 0
        return self.fact[n] * self.inv[n-r] % self.MOD

N, M = MAP()

mt = ModTools(M+1, MOD)
# とりあえずA単体としては順列全通り作れる
acnt = mt.nPr(M, N)

# Bについて、Aとの一致を避けるように作る
bcnt = 0
# Aと一致させる場所をi個決める、を0〜Nまで見て包除する
for i in range(N+1):
    # Aと確実に一致させる場所をN個からi個選ぶ * 残りをM-i個からN-i個並べる * 包除で+-
    bcnt += mt.nCr(N, i) * mt.nPr(M-i, N-i) * (-1)**i
    bcnt %= MOD
ans = acnt * bcnt % MOD
print(ans)
