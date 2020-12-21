# -*- coding: utf-8 -*-

"""
参考：https://www.hamayanhamayan.com/entry/2016/11/14/223727
　　　https://qiita.com/drken/items/03c7db44ccd27820ea0d
　　　https://kyopro-friends.hatenablog.com/entry/2019/01/12/231000
・期待値DP、メモ化再帰
・条件付き期待値：(期待値) = sum((条件付き期待値)*(その条件となる確率))
・式変形で遷移を作ってる人が多そうだけど、こっちの考え方で行くと、
　そもそも自己ループへの遷移を意識しなくて良さそう。
・300^3=2700万の再帰つらくてpypyでもTLE。。
"""

import sys
from collections import Counter

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
C = Counter(A)

# memo[i][j][k] := 1個残りがi個、2個残りがj個、3個残りがk個である状態にするまでの操作回数の期待値
memo = list3d(N+1, N+1, N+1, -1)
def rec(i, j, k):
    if memo[i][j][k] != -1:
        return memo[i][j][k]
    # 全て残0なら残り回数は0
    if i == j == k == 0:
        return 0
    # 今現在残っている皿を取る確率
    tmp = (i + j + k) / N
    # そのために必要な回数
    res = 1 / tmp
    # それ以降に必要な回数
    if i: res += (i/N) / tmp * rec(i-1, j, k)
    if j: res += (j/N) / tmp * rec(i+1, j-1, k)
    if k: res += (k/N) / tmp * rec(i, j+1, k-1)
    memo[i][j][k] = res
    return res

print(rec(C[1], C[2], C[3]))
