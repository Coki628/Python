# -*- coding: utf-8 -*-

"""
・numpy実装、python0.8秒AC。
・np.maximumで配列一括比較してループを1つ減らした。
・とはいえこれでもギリギリで、枝刈りのbreak入れてやっと通った感じ。。
・追加で2次元目の遷移の始点を調整した(1日目に3回使った状態とかを省いた)
　けど、速度ほぼ変わらず。1度に遷移させる数はあんま影響しないのかな。
"""

import sys
import numpy as np

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, M, K = MAP()
A = np.zeros(N+1, dtype=np.int64)
A[1:] = np.array(LIST(), dtype=np.int64)

# dp[i][j] := j日目にi回使った状態
INF = 10 ** 18
dp = np.full((M+1, N+1), -INF, dtype=np.int64)
dp[0,0] = 0
for i in range(M):
    end = N+1 - M + i+1
    # 1~K日後の範囲内で全ての日に遷移してみる
    for k in range(1, K+1):
        # 使用回数M回に届かない遷移はさせない
        if k == end: break
        # 各k日後への遷移をまとめてやる
        dp[i+1,i+k:end] = np.maximum(dp[i+1,i+k:end], dp[i,i:end-k]+A[i+k:end])

# M回終わって、あとK日で終わりまで行ければOK
ans = -INF
for i in range(N+1-K, N+1):
    ans = max(ans, dp[M,i])
if ans >= 0:
    print(ans)
else:
    print(-1)
