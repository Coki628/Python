# -*- coding: utf-8 -*-

"""
・500点自力AC！
・昨日のやつ(dwacon2020qual_b)と基本方針はほぼ同じ感じ。
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

class ModTools:
    """ 階乗たくさん使う時用のテーブル準備 """

    def __init__(self, MAX, MOD):
        """ MAX：階乗に使う数値の最大以上まで作る """
        
        MAX += 1
        self.MAX = MAX
        self.MOD = MOD
        # 階乗テーブル
        factorial = [1] * MAX
        factorial[0] = factorial[1] = 1
        for i in range(2, MAX):
            factorial[i] = factorial[i-1] * i % MOD
        # 階乗の逆元テーブル
        inverse = [1] * MAX
        # powに第三引数入れると冪乗のmod付計算を高速にやってくれる
        inverse[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
        for i in range(MAX-2, 0, -1):
            # 最後から戻っていくこのループならMAX回powするより処理が速い
            inverse[i] = inverse[i+1] * (i+1) % MOD
        self.fact = factorial
        self.inv = inverse
    
    def nCr(self, n, r):
        """ 組み合わせの数 (必要な階乗と逆元のテーブルを事前に作っておく) """

        if n < r: return 0
        # 10C7 = 10C3
        r = min(r, n-r)
        # 分子の計算
        numerator = self.fact[n]
        # 分母の計算
        denominator = self.inv[r] * self.inv[n-r] % self.MOD
        return numerator * denominator % self.MOD

N, K = MAP()
A = LIST()

mt = ModTools(N, MOD)
A.sort()
# 各要素が最小・最大として何回使われるか数える
mncnt = [0] * N
for i in range(N):
    mncnt[i] = mt.nCr(N-i-1, K-1)
mxcnt = mncnt[::-1]

mnsm = 0
mxsm = 0
for i in range(N):
    mnsm += A[i] * mncnt[i]
    mxsm += A[i] * mxcnt[i]
    mnsm %= MOD
    mxsm %= MOD
ans = mxsm - mnsm
ans %= MOD
print(ans)
