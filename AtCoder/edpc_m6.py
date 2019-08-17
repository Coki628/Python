# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/dp/submissions/6777059
・数え上げDP
・2次元+遷移先が多い系
・maspyさんのやつ見てnumpy実装作る。
・以下、得られた知見を列挙
　・その回に遷移させる区間の長さが一定なら、いもすっぽく準備して累積和すれば一撃
　・とにかく加減乗除が配列全体にまとめて当てれるのをうまく利用する。
　・np.cumsum(A, out=A)で戻り値の行き先を指定A(A = A.cumsum()でも同じぽいけど)
"""

import sys
import numpy as np

def input(): return sys.stdin.readline().strip()
def ceil(a, b=1): return int(-(-a // b))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, 1)
def gcd_list(nums): return reduce(gcd, nums, nums[0])
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, K = MAP()
A = LIST()

# dp[この子までの範囲で][飴を配る総数] = 場合の数
dp = np.zeros((N+1, K+1), dtype=np.int64)
# 0番目まで(誰もいない)の範囲で0個配る場合の数は1通り
dp[0,0] = 1

for i, a in enumerate(A):
    # 今回のiを次の遷移させる区間の始点としてまとめて追加
    dp[i+1] = dp[i]
    # 今回のiをa+1個ずらして、遷移させる区間の終点としてまとめて追加
    dp[i+1,a+1:] -= dp[i,:-(a+1)]
    # 累積和すれば、元の状態からそれぞれa個先まで遷移させた状態になる
    # (区間の始点・終点の記録→累積和の流れはいもす的な感じ)
    np.cumsum(dp[i+1], out=dp[i+1])
    dp[i+1] %= MOD

print(dp[N][K])
