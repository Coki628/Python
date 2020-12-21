# -*- coding: utf-8 -*-

"""
・自力AC
・1次元+定数添字のDP
・列を塗る際の制限は、隣の列と同じ色にならないことなので、
　直前列の色だけ状態として管理すればスッキリまとまる。
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

N = INT()
A = [[]] * 5
for i in range(5):
    A[i] = list(input())
# 行列反転
A = list(zip(*A))

# dp[i][j] := 旗のi列目まで見た時、直前列の色がj(0=r,1=b,2=w)の時の最小塗り替えマス数
dp = list2d(N+1, 3, INF)
for i in range(3):
    dp[0][i] = 0
for i, col in enumerate(A):
    # この列の元々の色を集計
    color = [0] * 3
    for c in col:
        if c == 'R':
            color[0] += 1
        elif c == 'B':
            color[1] += 1
        elif c == 'W':
            color[2] += 1
    for j in range(3):
        for k in range(3):
            # 直前列と同じ色にはできない
            if j == k:
                continue
            # 今回の列をk色に塗るのに必要なコスト(5 - 既にk色のマス)を足して遷移
            dp[i+1][k] = min(dp[i+1][k], dp[i][j] + (5 - color[k]))
ans = min(dp[N][0], dp[N][1], dp[N][2])
print(ans)
