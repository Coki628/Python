# -*- coding: utf-8 -*-

"""
・自力AC！
・ナップザップ*bitDP
・蟻本演習でも何回か実装した形。うまく組めたと思う。
・pypyはメモリ制限64Mが恐いので、DP配列は全部持たずに、直前と今の2つだけにしてある。
・計算量はN*2^M*M=5000*2^10*10=5100万、pypyAC0.3秒。
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

N, M, Y, Z = MAP()
ctob = {}
P = [0] * M
for i in range(M):
    c, p = input().split()
    P[i] = int(p)
    ctob[c] = i
B = [ctob[b] for b in input()]

# dp[i][S][j] := i番目のブロックまで考えて、取得済の色の集合がS、直前のブロックの色がjの時の最大得点
dp = list3d(2, 1<<M, M+1, -INF)
# 初期値：スタート時点では直前のブロック色はないので、存在しない色Mとする
dp[0][0][M] = 0
for i in range(N):
    cur = i % 2
    nxt = (i+1) % 2
    b = B[i]
    for S in range(1<<M):
        for j in range(M+1):
            # 今回のブロックは積まない
            dp[nxt][S][j] = max(dp[nxt][S][j], dp[cur][S][j])
            p = P[b]
            # 直前の色と一致していたら、ボーナスYを加える
            if b == j:
                p += Y
            # 今回のブロックを積む
            dp[nxt][S|1<<b][b] = max(dp[nxt][S|1<<b][b], dp[cur][S][j] + p)

for j in range(M+1):
    # 全色揃っている集合について、ボーナスZを加える
    dp[N%2][(1<<M)-1][j] += Z

ans = -INF
for S in range(1<<M):
    for j in range(M+1):
        ans = max(ans, dp[N%2][S][j])
print(ans)
