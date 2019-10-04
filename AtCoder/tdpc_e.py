# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-1、自力AC！
・桁DP
・Dの倍数の通り数 => mod Dで0になる数の通り数-1 の言い換え考察も大事だと思う。
・計算量はN*D*10=10000*100*10=1000万、pythonTLE、pypyAC0.7秒。
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

D = INT()
S = input()
N = len(S)

# dp0[i][j] := i桁目まで考えて、mod Dの中でjになる数の通り数(限定的な遷移)
dp0 = list2d(N+1, D, 0)
# dp1[i][j] := i桁目まで考えて、mod Dの中でjになる数の通り数(0~9まで全部遷移)
dp1 = list2d(N+1, D, 0)
dp0[0][0] = 1
for i in range(N):
    s = int(S[i])
    for j in range(D):
        for k in range(10):
            # 今回の桁kが整数Sの上からi桁目sより小さければ、
            # 次回以降はどこでも遷移できるように全部dp1に遷移させる
            if k < s:
                dp1[i+1][(j+k)%D] += dp0[i][j]
                dp1[i+1][(j+k)%D] += dp1[i][j]
                dp1[i+1][(j+k)%D] %= MOD
            # 同じだった状況が変わらないのでそのまま0は0、1は1へ
            elif k == s:
                dp0[i+1][(j+k)%D] += dp0[i][j]
                dp1[i+1][(j+k)%D] += dp1[i][j]
                dp0[i+1][(j+k)%D] %= MOD
                dp1[i+1][(j+k)%D] %= MOD
            # kが大きい場合はdp0は遷移させない
            else:
                dp1[i+1][(j+k)%D] += dp1[i][j]
                dp1[i+1][(j+k)%D] %= MOD
# 両方合わせる-1(0は倍数じゃないので)が答え
print((dp0[N][0]+dp1[N][0]-1)%MOD)
