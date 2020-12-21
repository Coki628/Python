# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/contest/1288/submission/68784436
・さらなる一撃解
・実はN種類から重複有でM*2個選ぶ重複組み合わせだけでいい。これどうやって思いつくんだろ。
・ちょっと分かったかも。例えば、
　N=2,M=2として、A[0]=1ならA[1]は1以上N以下、A[1]=1ならB[1]は1以上N以下、B[1]=1ならB[0]は1以上N以下、
　って感じでAの先頭から末尾〜Bの末尾から先頭を見ていくと、長さM*2の1つの数列に対して
　広義単調増加列の数え上げやってるだけみたいに出来てる。
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

    def nHr(self, n, r):

        return self.nCr(r+n-1, r)

N, M = MAP()

mt = ModTools(N+M*2, MOD)
ans = mt.nHr(N, M*2)
print(ans)
