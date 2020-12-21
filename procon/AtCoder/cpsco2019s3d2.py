# -*- coding: utf-8 -*-

"""
・貪欲から頭を切り替えてDPっぽい遷移を考えてみる。
・桁DPっぽく、1文字ずつ遷移できるところに遷移させていく。
・これで400点自力AC。時間かかったけどね。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(li): return reduce(lcm, li, 1)
def gcd_list(li): return reduce(gcd, li, 0)
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

N = INT()
S = input()

# dp[i][j] := i+1文字目に、文字jにできるかどうか
dp = [{'R': False, 'G': False, 'B': False} for i in range(N)]
# 最初はRしかダメ
dp[0]['R'] = True

# 桁DPっぽく、1文字ずつ遷移できるところに遷移させていく
for i in range(N-1):
    # Rだった場合の遷移
    if S[i] == 'R':
        # R,Gへの遷移は、最後から2,1文字目以降にはできない
        dp[i+1]['R'] = dp[i]['R'] and i < N-3
        dp[i+1]['G'] = dp[i]['R'] and i < N-2
    # Gだった場合の遷移
    elif S[i] == 'G':
        dp[i+1]['R'] = dp[i]['G'] and i < N-3
        dp[i+1]['B'] = dp[i]['G']
    # Bだった場合の遷移
    else:
        dp[i+1]['R'] = dp[i]['B'] and i < N-3
        dp[i+1]['G'] = dp[i]['B'] and i < N-2
        dp[i+1]['B'] = dp[i]['B']

# N文字終わって辿りつけていればOK
if dp[N-1][S[-1]]:
    Yes()
else:
    No()
