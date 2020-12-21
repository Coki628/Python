# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-5、自力AC！
・部分和DP応用、要素の値に必要な情報を持たせる。
・pythonTLEは想定内だったけど、最初pypyがMLEした。
・原因はfloat('inf')で初期化した配列。これを10**18にしたらAC。
・DP配列のサイズがでかい時(今回10000*1000で1000万ある)はfloat('inf')で作らない方が良さそうだ。
・計算量1000万でpythonTLE、pypyAC0.9秒。
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

N, M, L, X = MAP()
A = LIST()

# dp[i][j] := i個目の燃料まで考えた時、位置jに止まるための最小の周回数
dp = list2d(N+1, M, INF)
dp[0][0] = 0
for i in range(N):
    a = A[i]
    for j in range(M):
        # X周以上するとNGなのでそれ以降は遷移させる必要ない
        if dp[i][j] < X:
            # この燃料を使わない遷移
            dp[i+1][j] = min(dp[i+1][j], dp[i][j])
            # 使う遷移
            if j + a < M:
                # 今回の周回に収まるなら普通に遷移
                dp[i+1][j+a] = min(dp[i+1][j+a], dp[i][j])
            else:
                # 超過する周回数を追加して遷移
                b = (j+a) // M
                dp[i+1][(j+a)%M] = min(dp[i+1][(j+a)%M], dp[i][j]+b)
if dp[N][L] < X:
    Yes()
else:
    No()
