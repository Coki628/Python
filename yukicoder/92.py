"""
・結構すんなり自力AC
・グラフ、DP
・よくある最短経路求めるのとはちょっと違う。制約が小さいのでDPした。
"""

import sys
from collections import defaultdict

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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N, M, K = MAP()
nodes = defaultdict(set)
for i in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    nodes[c].add((a, b))
    nodes[c].add((b, a))
A = LIST()

# dp[i][j] := i個目の経路まで見て、最後に街jにいる状態が可能かどうか
dp = list2d(K+1, N, 0)
# 最初は全ての街に可能性がある
for i in range(N):
    dp[0][i] = 1
for i, a in enumerate(A):
    for j in range(N):
        for k in range(N):
            # コストaな経路の集合に街(j, k)を繋ぐものがあれば遷移できる
            if (j, k) in nodes[a]:
                dp[i+1][k] |= dp[i][j]
ans = []
for i in range(N):
    # いる可能性のある街を答えに追加
    if dp[K][i]:
        ans.append(i+1)
print(len(ans))
print(*ans)
