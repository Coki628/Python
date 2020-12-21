# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC
・文字列DP(って言っていいのか？)
・文字列スライスが配列外の添字指すとバグるか心配だったけど、
　確認したら、大きく切ろうとした分は、末尾までになるだけだったので大丈夫。
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

S = input()
N = len(S)

# dp[i] := i文字目までをS=Tとすることができるか
dp = [0] * (N + 9)
dp[0] = 1
for i in range(N):
    if dp[i]:
        if S[i:i+5] == 'dream':
            dp[i+5] = 1
        if S[i:i+7] == 'dreamer':
            dp[i+7] = 1
        if S[i:i+5] == 'erase':
            dp[i+5] = 1
        if S[i:i+6] == 'eraser':
            dp[i+6] = 1
if dp[N]:
    YES()
else:
    NO()
