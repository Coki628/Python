# -*- coding: utf-8 -*-

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, Ma, Mb = map(int, input().split())
l = [[0] * 3 for i in range(N)]
for i in range(N):
    l[i][0], l[i][1], l[i][2] = map(int, input().split())

ans = INF
for i in range(N):
    sm = [0, 0, 0]
    for j in range(i, N):
        sm[0] += l[j][0]
        sm[1] += l[j][1]
        sm[2] += l[j][2]
        if sm[0] / sm[1] == Ma / Mb:
            ans = min(sm[2], ans)

if ans == INF:
    print(-1)
else:
    print(ans)
