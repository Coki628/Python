# -*- coding: utf-8 -*-

"""
・蟻本演習、自力AC！
・いわゆる1方向系、ナップザップ系DPの応用。2つめ以降の添字が多い。
・でもその前準備の、1〜6を素数冪でまとめる、目的の数より大きい部分は上限値のところにまとめる、
　みたいな考察部分のが大事な気もする。
"""

import sys
from collections import defaultdict

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
INF = float('inf')
MOD = 10 ** 9 + 7

def factorize(num: int) -> dict:
    """ 素因数分解 """
    from math import sqrt
    from collections import Counter

    d = Counter()
    # 終点はルート切り捨て+1
    for i in range(2, int(sqrt(num))+1):
        # 今回は2,3,5以外があったら即終了
        if i == 6:
            print(0)
            exit()
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

N, D = MAP()

d = factorize(D)
need2 = d[2]
need3 = d[3]
need5 = d[5]

# dp[i][(j, k, l)] := サイコロをi回振って、出た目の積の素数冪が2^j,3^k,5^lの状態に遷移する確率
dp = [defaultdict(int) for i in range(N+1)]
# 初期化：0回振って全部0の確率は100%(そりゃそうだ)
dp[0][(0, 0, 0)] = 1
for i in range(N):
    for k, v in dp[i].items():
        cnt2, cnt3, cnt5 = k
        # 今回の目が1〜6の場合をそれぞれ遷移(必要数をオーバーした分は同一とみなす)
        dp[i+1][(cnt2, cnt3, cnt5)] += v / 6
        dp[i+1][(min(need2, cnt2+1), cnt3, cnt5)] += v / 6
        dp[i+1][(cnt2, min(need3, cnt3+1), cnt5)] += v / 6
        dp[i+1][(min(need2, cnt2+2), cnt3, cnt5)] += v / 6
        dp[i+1][(cnt2, cnt3, min(need5, cnt5+1))] += v / 6
        dp[i+1][(min(need2, cnt2+1), min(need3, cnt3+1), cnt5)] += v / 6

# 素数冪で必要な個数の2,3,5が揃っている確率
print(dp[N][(need2, need3, need5)])
