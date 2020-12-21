# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/momohara_kyopro/status/1237054475392520197
　　　https://twitter.com/laycrs/status/1237054357918470144
・別解
・数え上げ
・最大値を固定して考える。こっちのがまだ導きやすいかもしれない。
・お気持ちはソース内コメント参照。
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 998244353

class ModTools:

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
        for i in range(MAX-2, 0, -1):
            inverse[i] = inverse[i+1] * (i+1) % MOD
        self.fact = factorial
        self.inv = inverse
    
    def nCr(self, n, r):

        if n < r: return 0
        r = min(r, n-r)
        numerator = self.fact[n]
        denominator = self.inv[r] * self.inv[n-r] % self.MOD
        return numerator * denominator % self.MOD

N, M = MAP()

mt = ModTools(M, MOD)
if N == 2:
    print(0)
    exit()

ans = 0
# 最大値をkに固定する
for k in range(M, N-2, -1):
    # kより小さいk-1種類からN-2個を選ぶ * ダブりをN-2種類から1つ選ぶ * 最大値とダブり以外のN-3個を左右どちらかに振る
    ans += mt.nCr(k-1, N-2) * (N-2) * pow(2, N-3, MOD) % MOD
    ans %= MOD
print(ans)
