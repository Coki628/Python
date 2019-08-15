# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc003
　　　http://tutuz.hateblo.jp/entry/2018/08/23/231229
　　　http://tkori.hateblo.jp/entry/2015/12/16/180521
・包徐原理：奇数個の共通領域を足して、偶数個の共通領域を引く。
・自力実装から改善1
・まず選び出す所で空白じゃなくてデスクとラックを選んだら、全部同じにできた。
　確かにそうだ。何でわざわざ空白の方数えたのか。。
・あと変数名ちょっと整理。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

class FactInvMOD:
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
        # あり得ない状況が来たら0を返す
        if n < 0 or r < 0: return 0
        # 10C7 = 10C3
        r = min(r, n-r)
        # 分子の計算
        numerator = self.fact[n]
        # 分母の計算
        denominator = self.inv[r] * self.inv[n-r] % self.MOD
        return numerator * denominator % self.MOD

    def nPr(self, n, r):
        """ 順列 """
        if n < r: return 0
        return self.fact[n] * self.inv[n-r] % self.MOD

    def nHr(self, n, r):
        """ 重複組み合わせ """
        # r個選ぶところにN-1個の仕切りを入れる
        return self.nCr(r+n-1, r)

R, C = MAP()
X, Y = MAP()
D, L = MAP()

fim = FactInvMOD(X*Y, MOD)

# 区画内でのデスク、ラック、空白の配置の全パターン数
total = fim.nCr(X*Y, D+L) * fim.nCr(D+L, D) % MOD

# いずれかの辺が真っ白ならNG
# 横1辺
W1 = fim.nCr(X*(Y-1), D+L) * fim.nCr(D+L, D) % MOD
# 縦1辺
H1 = fim.nCr((X-1)*Y, D+L) * fim.nCr(D+L, D) % MOD
# 横2辺
W2 = fim.nCr(X*(Y-2), D+L) * fim.nCr(D+L, D) % MOD if Y >= 2 else 0
# 縦2辺
H2 = fim.nCr((X-2)*Y, D+L) * fim.nCr(D+L, D) % MOD if X >= 2 else 0
# 縦横1辺ずつ
W1H1 = fim.nCr((X-1)*(Y-1), D+L) * fim.nCr(D+L, D) % MOD
# 横2辺、縦1辺
W2H1 = fim.nCr((X-1)*(Y-2), D+L) * fim.nCr(D+L, D) % MOD if Y >= 2 else 0
# 横1辺、縦2辺
W1H2 = fim.nCr((X-2)*(Y-1), D+L) * fim.nCr(D+L, D) % MOD if X >= 2 else 0
# 4辺全部
W2H2 = fim.nCr((X-2)*(Y-2), D+L) * fim.nCr(D+L, D) % MOD if X >= 2 and Y >= 2 else 0

# 包徐原理でNGなパターンの通り数を出す
ng = (W1*2 + H1*2 - W2 - H2 - W1H1*4 + W2H1*2 + W1H2*2 - W2H2) % MOD
ok = (total - ng) % MOD

# 区画内での配置の通り数 * 部屋全体から区画を配置する通り数
print(ok*(R-X+1)*(C-Y+1)%MOD)
