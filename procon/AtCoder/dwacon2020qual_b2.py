# -*- coding: utf-8 -*-

"""
・600点自力AC！参戦中のこれは嬉しい。
・問題の見た目は期待値だけど、実際は回数数えて総和取る感じ。
・N=8くらいまで実験して、規則的に増加するのを見出だせた。
・pythonAC0.6秒、pypyAC0.3秒。
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

    def div(self, x, y):
        """ MOD除算 """

        return x * pow(y, self.MOD-2, self.MOD) % self.MOD

N = INT()
A = [0] + LIST()
B = [A[i+1] - A[i] for i in range(N)]

mt = ModTools(N, MOD)
add = mt.fact[N-1]
# スライムが区間iを通る合計回数をそれぞれ求める
cnt = [0] * N
for i in range(1, N):
    cnt[i] = cnt[i-1] + add
    cnt[i] %= MOD
    add *= mt.div(i ,i+1)
    add %= MOD

ans = 0
for i in range(1, N):
    # 区間長を掛ければ合計距離
    ans += B[i] * cnt[i]
    ans %= MOD
print(ans)
