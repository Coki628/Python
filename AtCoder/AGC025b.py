# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/agc025/editorial.pdf
・何とか3色を2色として扱うんだろうなあとは考えつつも方法は出ず。
・数え上げ、組み合わせ、式変形
・A+Bの色は、A,B両方で塗るとみなすと、A,B色だけの問題にできる。
・片方決め打ちでループを1つ減らす。=Kを式変形することで一意にできる。
・A,Bそれぞれいくつ使うかが決まれば、後はN個から選ぶ組み合わせ。
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
MOD = 998244353

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

N, A, B, K = MAP()

ans = 0
mt = ModTools(N, MOD)
for x in range(N+1):
    # 式変形：A*x + B*y = K → y = (K -  A*x) / B
    y = (K -  A*x) / B
    if y < 0 or not y.is_integer():
        continue
    y = int(y)
    ans += mt.nCr(N, x) * mt.nCr(N, y)
    ans %= MOD
print(ans)
