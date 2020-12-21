# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/pakencamp-2018-day3/editorial.pdf
・後ろから貪欲
・Si(今回の最大) < Ai+1(次の値) だと、後ろから貪欲に取ってある値を達成可能か判定できる。
　(なんでそれで大丈夫かは自分ノート参照)
・計算量は10万クエリ*最大約88が3回(実際はもっと減るだろうけど)で約2640万、pythonTLEでpypyAC1.4秒。
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

N = 88

# 各回の鍵盤の値を取得
def get_piano(a, b):
    dp = [0] * N
    dp[0] = a
    dp[1] = b
    for i in range(2, N):
        dp[i] = dp[i-1] + dp[i-2]
    return dp

# 最初以外は後ろから貪欲で処理できる
def get_sum(A, x):
    res = 0

    # A[0]を使わない場合
    cnt = 0
    i = N-1
    while i > 0:
        if cnt+A[i] <= x:
            cnt += A[i]
            i -= 1
        i -= 1
    if cnt == x:
        res += 1

    # A[0]を使う場合
    cnt = A[0]
    i = N-1
    while i > 1:
        if cnt+A[i] <= x:
            cnt += A[i]
            i -= 1
        i -= 1
    if cnt == x:
        res += 1

    return res

for _ in range(INT()):
    a, b, x = MAP()
    piano = get_piano(a, b)
    res = get_sum(piano, x)
    print(res)
