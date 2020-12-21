# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/iroha2019-day2/editorial-E.pdf
　　　https://www.risuuhakusyo.com/chouhuku-kumiawase
・組み合わせ数え上げ系
・MOD事前テーブル準備
・重複組み合わせ
　(選ぶ物の数+仕切り)C(選ぶ物の数)
・考察大変。詳細は参考で。
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

def init_fact_inv(MAX: int, MOD: int) -> list:
    """
    階乗たくさん使う時用のテーブル準備
    MAX：階乗に使う数値の最大以上まで作る
    """
    MAX += 1
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
    return factorial, inverse

# 組み合わせの数(必要な階乗と逆元のテーブルを事前に作っておく)
def nCr(n, r, factorial, inverse):
    if n < r or r < 0 or n < 0: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorial[n]
    # 分母の計算
    denominator = inverse[r] * inverse[n-r] % MOD
    return numerator * denominator % MOD

N,M=MAP()

fact,inv=init_fact_inv(N+M+1, MOD)

sm=0
for k in range(1, min(N, M)+1):
    # A：各要素が1か2(0か1と考える)で長さkの列で、総和がNになるものの数
    cnta=nCr(k, N-k, fact, inv)
    # B：各要素が1~M(0~M-1と考える)で長さkの列で、総和がMになるものの数(重複組み合わせ)
    cntb=nCr(M-1, M-k, fact, inv)
    sm+=cnta*cntb
    sm%=MOD
ans=(nCr(N+M-2, N-1, fact, inv)-sm)%MOD
print(ans)
