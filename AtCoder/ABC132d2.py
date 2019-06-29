# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc132/editorial.pdf
　　　https://www.youtube.com/watch?v=mso8tE1yMl8
・組み合わせ系
・重複組み合わせ
・仕切り、切り分けることでうまくグループを作る。
・後半の計算をPDFの方にしてみた。
・nCr(R+1, i)
　R+1箇所の赤の間(両端含む)のうち、どこに青のグループを置くか
・nCr(B-1, i-1)
　B-1箇所の青の間(両端なし)のうち、i-1箇所で切ればi個のグループになる。
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

def init_fact_inv(MAX: int):
    """ 階乗たくさん使う時用のテーブル準備

    Parameters
    ----------
        MAX：階乗に使う数値の最大以上まで作る
    Returns
    -------
        factorial (list<int>), inverse (list<int>)
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

def nCr(n, r, factorial, inverse):
    """ 組み合わせの数 (必要な階乗と逆元のテーブルを事前に作っておく) """
    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorial[n]
    # 分母の計算
    denominator = inverse[r] * inverse[n-r] % MOD
    return numerator * denominator % MOD

N,K=MAP()
R=N-K
B=K

fact, inv = init_fact_inv(N)

for i in range(1, K+1):
    print(nCr(R+1, i, fact, inv) * nCr(B-1, i-1, fact, inv) % MOD)
