# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-16
・bitDP、部分点2想定、がしかし部分点2が取れず。。
・部分点2の計算量、2^30*30*10。。2^30だけでも10億ある、、ダメだこりゃ。
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
INF = float('inf')
MOD = 10007

N = INT()
A = LIST()

# 国毎ではなく人毎で国を値に持ったリストを作っておく
B = []
for i, a in enumerate(A):
    for _ in range(a):
        B.append(i)
M = len(B)

# dp[bit][j] := 席順確定済の人の集合をbit、直前に追加した人の国をjとして、その状態になる通り数
dp = list2d(1<<M, N, 0)
# 初期化：最初の1人はどの国の人でも確実に追加できる
for i, b in enumerate(B):
    dp[1<<i][b] = 1

for bit in range(1, 1<<M):
    # 追加する人ループ
    for i, b in enumerate(B):
        # この集合に既に含まれている人はスキップ
        if (bit>>i) & 1:
            continue
        # 直前の国ループ
        for prev in range(N):
            # 自国or隣国でなければ遷移させる
            if abs(b-prev) >= 2:
                dp[bit|(1<<i)][b] += dp[bit][prev]
                dp[bit|(1<<i)][b] %= MOD 

print(sum(dp[-1])%MOD)
