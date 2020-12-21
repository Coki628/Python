# -*- coding: utf-8 -*-

"""
参考：https://kimiyuki.net/writeup/algo/atcoder/tdpc-j/
・蟻本演習3-4-2
・bitDP、期待値DP
・先に前計算で全状態の部分期待値を出しておく方針なら、
　計算中にはどこにボールがあるのかないのかは意識しなくて済む。
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

N = INT()
A = LIST()
MAX = 16

# dp[S] := まだ倒していないピンの集合Sの状態から始めて、残る全てのピンが倒れるまでに投げる回数の期待値の最小値
dp = [INF] * (1<<MAX)
dp[0] = 0
for S in range(1, 1<<MAX):
    # ボールを投げる各位置ついて試して、今の集合の状態になれる最小を取る
    for i in range(1, MAX-1):
        tmp = 0
        for j in range(i-1, i+2):
            if S & (1<<j):
                tmp += 1 / 3
        if tmp == 0:
            continue
        # まだ倒していないピンを倒すまでに投げる回数
        nxt = 1 / tmp
        # 部分集合から必要な値を持ってきて足す
        for j in range(i-1, i+2):
            if S & (1<<j):
                # それ以降に投げる必要のある回数
                nxt += (1/3) / tmp * dp[S^(1<<j)]
        dp[S] = min(dp[S], nxt)

ans = 0
for a in A:
    ans |= 1<<a
print(dp[ans])
