# -*- coding: utf-8 -*-

"""
参考：https://simezi-tan.hatenadiary.org/entry/20140910/1410309068
・蟻本演習3-4-2
・bitDP、期待値DP、メモ化再帰
・練習のために、メモ化再帰版も実装。
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

# memo[S] := 既に倒したピンの集合Sの状態から始めて、残る全てのピンが倒れるまでに投げる回数の期待値の最小値
memo = [-1] * (1<<MAX)
def rec(S):
    if memo[S] != -1:
        return memo[S]
    # 全てのピンを倒したらその先に投げる回数は0
    if S == (1<<MAX)-1:
        return 0
    res = INF
    # ボールを投げる各位置ついて試して、今の集合の状態になれる最小を取る
    for i in range(1, MAX-1):
        tmp = 0
        for j in range(i-1, i+2):
            if not S & (1<<j):
                tmp += 1 / 3
        if tmp == 0:
            continue
        # まだ倒していないピンを倒すまでに投げる回数
        nxt = 1 / tmp 
        # 今の集合に含まれない部分について再帰的に調べていく
        for j in range(i-1, i+2):
            if not S & (1<<j):
                # それ以降に投げる必要のある回数
                nxt += (1/3) / tmp * rec(S|(1<<j))
        res = min(res, nxt)
    memo[S] = res
    return res

ans = (1<<MAX) - 1
for a in A:
    # 実際にボールがある各位置についてビットを0にしておく
    ans ^= 1<<a
print(rec(ans))
