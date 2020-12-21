# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-16
・bitDP、練習なのでTLE想定。
・テストケース少なすぎて合ってるか確証が持てない。
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
MOD = 10 ** 9 + 7

def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

A = LIST()
N = 26

# 文字の種類毎ではなく各文字毎で種類を値に持ったリストを作っておく
B = []
for i, a in enumerate(A):
    for _ in range(a):
        B.append(i)
M = len(B)

# dp[bit][j] := 配置位置済の文字の集合をbit、直前に追加した文字の種類をjとして、その状態になる通り数
dp = list2d(1<<M, N, 0)
# 初期化：最初の1文字目はどの種類の文字でも確実に追加できる
for i, b in enumerate(B):
    dp[1<<i][b] = 1

for bit in range(1, 1<<M):
    # 追加する文字ループ
    for i, b in enumerate(B):
        # この集合に既に含まれている文字はスキップ
        if (bit>>i) & 1:
            continue
        # 直前の文字の種類ループ
        for prev in range(N):
            # 同じ文字でなければ遷移させる
            if b != prev:
                dp[bit|(1<<i)][b] += dp[bit][prev]
                dp[bit|(1<<i)][b] %= MOD

ans = sum(dp[-1]) % MOD
for a in A:
    # 同じ文字の重複分を除く
    if a >= 2:
        ans //= a
print(ans)
