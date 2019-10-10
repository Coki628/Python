# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-1、自力AC
・ナップザック系1方向DPで2次元+もう1ループの3重のやつ。
・計算量200*200*100=400万はきついかと思ったけどpythonで0.8秒AC。
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

D, N = MAP()
T = [0] + LIST(D)
ABC = []
for i in range(N):
    a, b, c = MAP()
    ABC.append((a, b, c))
_, _, C = zip(*ABC)
mxC = max(C)

# dp[i][j] := i日目まで見て、前日の派手さがjだった場合の最大値
dp = list2d(D+1, mxC+1, -INF)
# 初期化：初日を終わらせる
t = T[1]
for i in range(N):
    a, b, c = ABC[i]
    # 気温の範囲内の服だけ選べる
    if a <= t <= b:
        dp[1][c] = 0

for i in range(1, D):
    t = T[i+1]
    for prevc in range(mxC+1):
        # ありえない状態からは遷移させない
        if dp[i][prevc] != -INF:
            for j in range(N):
                a, b, c = ABC[j]
                if a <= t <= b:
                    dp[i+1][c] = max(dp[i+1][c], dp[i][prevc]+abs(c-prevc))
print(max(dp[D]))
