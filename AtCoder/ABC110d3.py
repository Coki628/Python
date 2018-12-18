# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2018/09/23/224100
　　　http://tutuz.hateblo.jp/entry/2018/09/24/121248
・素因数分解と重複組み合わせ
・毎回階乗やると死ぬから階乗と逆元のテーブル作る(忘れてた)
"""

from collections import defaultdict
from math import sqrt

MOD = 10 ** 9 + 7

def fact_prime(num):
    d = defaultdict(int)
    # 終点はルート切り捨て+1
    end = int(sqrt(num)) + 1
    for i in range(2, end+1):
        cnt = 0
        # 素因数分解：小さい方から割れるだけ割って素数をカウント
        while num % i == 0:
            num //= i
            d[i] += 1
        # 1まで分解したら終了
        if num == 1:
            break
    # 最後に残ったnumは素数(ただし1^1は1^0なので数に入れない)
    if num != 1:
        d[num] += 1
    return d

# とりあえずv+N-1が収まればいいはず
MAX = 10 ** 5 * 2
# 予め組み合わせ計算に必要な階乗と逆元のテーブルを作っておく
factorial = [1] * (MAX)
factorial[0] = factorial[1] = 1
for i in range(2, MAX):
    factorial[i] = factorial[i-1] * i % MOD

inverse = [1] * (MAX)
# powに第三引数入れると冪乗のmod付計算を高速にやってくれる
inverse[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
for i in range(MAX-2, 0, -1):
    # 最後から戻っていくこのループならMAX回powするより処理が速い
    inverse[i] = inverse[i+1] * (i+1) % MOD

# 組み合わせの数
def nCr(n, r):
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorial[n]
    # 分母の計算
    denominator = inverse[r] * inverse[n-r] % MOD
    return numerator * denominator % MOD

N, M = map(int, input().split())

ans = 1
for k, v in fact_prime(M).items():
    # 重複組み合わせ
    # v個のkとN-1個の仕切りから、v個を並べる(v+N-1個のマスからv個を選び出す組み合わせ)
    ans = (ans * nCr(v+N-1, v)) % MOD
print(ans)
