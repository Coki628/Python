# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC！しょーもないとこ(初期化-INFじゃなくて0にしてた)で最後ちょっと詰まったけど、方針としてはすぐ出た。
・ナップザック系DP、ただし各値を+とみなすか-とみなすかで2^3通りを見る。
・でもこれDPいらなかった。2^3通り考えるの考察さえ出てれば、ソートして貪欲でよかった。
・とはいえ昔の自分は当時のコメントを見る限り、DPじゃ無理だと思い込んだみたいなので、
　そこにリベンジを果たすことが出来てよかったかな。
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

N, M = MAP()
XYZ = []
for i in range(N):
    x, y, z = MAP()
    XYZ.append((x, y, z))

# dp[i][j][S] := i個目まで見て、j個選んだ状態で、3つの値をそれぞれ+とみなすか-とみなすかの2^3通りを集合Sで表した時の最大値
dp = list3d(N+1, M+1, 1<<3, -INF)
# 初期化：出発点となる2^3箇所は0で他は-INF
for S in range(1<<3):
    dp[0][0][S] = 0
for i in range(N):
    for j in range(M+1):
        # 枝刈り
        if i < j:
            break
        for S in range(1<<3):
            # このケーキを選ばない遷移
            dp[i+1][j][S] = max(dp[i+1][j][S], dp[i][j][S])
            if j < M:
                nxt = 0
                for k in range(3):
                    if S & 1<<k:
                        # この値は+とみなす
                        nxt += XYZ[i][k]
                    else:
                        # この値は-とみなす
                        nxt += -XYZ[i][k]
                # このケーキを選ぶ遷移
                dp[i+1][j+1][S] = max(dp[i+1][j+1][S], dp[i][j][S] + nxt)
# N個見てM個選び終わって、+-をどう扱ったかの2^3通りから最大を取る
print(max(dp[N][M]))
