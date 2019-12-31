# -*- coding: utf-8 -*-

"""
・400点自力AC！
・添字2で1つ前を覚えておくDP
・そのままだとK個前だけど、mod Kで分ければ全部1つ前になる。
・計算量は10万に添字2と遷移の3*3で90万。pythonAC1.1秒、pypyAC0.4秒。
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

N, K = MAP()
R, S, P = MAP()
T = list(input())

# mod KでK個のグループに分ける
T2 = [[] for i in range(K)]
for i, t in enumerate(T):
    T2[i%K].append(t)

ans = 0
for k in range(K):
    M = len(T2[k])
    # dp[i][j] := i回目まで見て、前回出した手がr,s,p(j=0,1,2)だった時の最大スコア
    dp = list2d(M+1, 3, 0)
    for i, t in enumerate(T2[k]):
        # 前回出した手j
        for j in range(3):
            # 今回出す手l
            for l in range(3):
                # 初回以外で、前回と同じ手なら遷移不可
                if i != 0 and j == l:
                    continue
                nxt = 0
                # 相手がrならp(2)の時に得点
                if t == 'r' and l == 2:
                    nxt = P
                # 相手がsならr(0)の時に得点
                elif t == 's' and l == 0:
                    nxt = R
                # 相手がpならs(1)の時に得点
                elif t == 'p' and l == 1:
                    nxt = S
                dp[i+1][l] = max(dp[i+1][l], dp[i][j] + nxt)
    # このグループでのスコアを合計に追加
    ans += max(dp[M])
print(ans)
