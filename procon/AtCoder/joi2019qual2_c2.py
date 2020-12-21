# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/joi2020yo2/submissions/9413802
・桁和の計算、文字列に変換しないやつにした。
・さすがに100万回もやるとこっちのがだいぶ速い。
　pythonAC1.8→1.2秒、pypyAC0.7→0.4秒。
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

def digit_sum(n):
    """ 各桁の和を求める：O(logN) """

    ans = 0
    while n > 0:
        ans += n % 10
        n //= 10
    return ans

N = INT()

# dp[i] := x + 桁和(x) を繰り返して、iになりうる通り数
dp = [1] * (N+1)
for i in range(1, N):
    nxt = i + digit_sum(i)
    if nxt <= N:
        dp[nxt] += dp[i]
print(dp[N])
