# -*- coding: utf-8 -*-

"""
・400点自力AC、それなりに時間かかったけど。
・最初N=20万だし、1重ループでnCrやる系かと思って色々考えたけどうまくいかず、
　最終的にDPでいけるに気付けて良かった。
・貰うDP、数え上げDP
・普通にやったら状態数Nの各遷移もN回でN^2、N=20万なのでダメ。
・遷移まとめられそうかな、と思ったらできた。
・累積和で前計算して、貰う遷移を高速化。
・該当位置の取得には二分探索。
・python0.9秒、pypy0.7秒AC。
・ちなみにaccumulate使わないでforループで都度MOD取ってみたところ、
　むしろちょっと遅くなった。accumulate有能だな。。
"""

import sys
from itertools import accumulate
from bisect import bisect_left

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

N, M = MAP()
A = LIST()
B = LIST()

A.sort()
B.sort()
# dp1[i][j] := 1,4人目についての通り数
dp1 = list2d(2, M, 0)
# dp2[i][j] := 2,3人目についての通り数
dp2 = list2d(2, N, 0)
# 初期値：最初の1人は誰でもOKなので全部1
for i in range(M):
    dp1[0][i] = 1
# 累積和して、まとめて貰うDP遷移をする
dp1[0] = list(accumulate(dp1[0]))

# 2人目
for i in range(N):
    # どこから取れるかは二分探索
    idx = bisect_left(B, A[i]) - 1
    if idx >= 0:
        dp2[0][i] = dp1[0][idx] % MOD
dp2[0] = list(accumulate(dp2[0]))
# 3人目
for i in range(N):
    idx = bisect_left(A, A[i]) - 1
    if idx >= 0:
        dp2[1][i] = dp2[0][idx] % MOD
dp2[1] = list(accumulate(dp2[1]))
# 4人目
for i in range(M):
    idx = bisect_left(A, B[i]) - 1
    if idx >= 0:
        dp1[1][i] = dp2[1][idx] % MOD

print(sum(dp1[1])%MOD)
