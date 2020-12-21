# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・なんとか自力AC
・ゲームDP、メモ化再帰
・必死で考えて添字を1つ減らした。
　未証明なまま、これprevいらないんじゃね？ってやっただけなので、全然いけてはいない。
・で、どうして大丈夫か考えた。
　prev使うのは決着の時だけなので、持ち回る必要はあってもメモに入れる必要はない。
　メモ的にはcurの値がその時点での相手方の持ってる番号になってる。
　むしろ工夫すれば、持ち回る必要もない。(前回自分が解いた時のやつ参照)
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

N, Z, W = MAP()
A = [W] + LIST()

memo = list2d(N+1, 2, -1)
def rec(cur, prev, turn):
    if cur == N:
        return abs(A[cur] - A[prev])
    if memo[cur][turn] != -1:
        return memo[cur][turn]
    
    if turn == 0:
        res = 0
        for nxt in range(cur+1, N+1):
            res = max(res, rec(nxt, cur, 1 - turn))
    else:
        res = INF
        for nxt in range(cur+1, N+1):
            res = min(res, rec(nxt, cur, 1 - turn))
    memo[cur][turn] = res
    return res
print(rec(0, 0, 0))
