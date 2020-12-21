# -*- coding: utf-8 -*-

"""
・メモ化再帰
・このままだとWA
・メモなくすとTLE
"""
 
import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7
 
N, Ma, Mb = map(int, input().split())
l = [[0] * 3 for i in range(N)]
for i in range(N):
    l[i][0], l[i][1], l[i][2] = map(int, input().split())
 
memo = [[[-1] * 401  for i in range(401)] for i in range(N+1)]
def dfs(cur, a, b, c):
    if memo[cur][a][b] != -1:
        return memo[cur][a][b]
    mn = INF
    if b != 0 and a / b == Ma / Mb:
        mn = min(c, mn)
    if cur == N:
        return mn
    mn = min(dfs(cur+1, a+l[cur][0], b+l[cur][1], c+l[cur][2]), mn)
    mn = min(dfs(cur+1, a, b, c), mn)
    memo[cur][a][b] = mn
    return mn
 
ans = dfs(0, 0, 0, 0)
if ans == INF:
    print(-1)
else:
    print(ans)
