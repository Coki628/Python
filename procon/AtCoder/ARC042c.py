# -*- coding: utf-8 -*-

"""
・自力AC！
・考察すんなり進んでいい感じだった。
・ソートしてナップザックに帰着
・計算量5000^2=2500万でpypyAC1.6秒。
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

N, P = MAP()
AB = []
for i in range(N):
    a, b = MAP()
    AB.append((a, b))

# タダ扱いになるのが最安のおやつになるようにソートしておく
AB.sort()

# dp[i][j][k] := i個目のおやつまで見て、使った金額がjで1つ目のおやつをまだ取ってないor取った(k=0,1)状態での満足度の最大値
dp = list3d(2, P+1, 2, 0)
for i in range(N):
    cur = i % 2
    nxt = (i+1) % 2
    a, b = AB[i]
    for j in range(P+1):
        # このおやつは取らない
        dp[nxt][j][1] = max(dp[nxt][j][1], dp[cur][j][1])
        # 1つ目のおやつとして取る
        dp[nxt][j][1] = max(dp[nxt][j][1], dp[cur][j][0] + b)
        # 2つ目以降として取る
        if j+a <= P:
            dp[nxt][j+a][1] = max(dp[nxt][j+a][1], dp[cur][j][1] + b)
print(dp[N%2][P][1])
