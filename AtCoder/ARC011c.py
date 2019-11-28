# -*- coding: utf-8 -*-

"""
・自力AC！
・グラフに帰着、BFS、経路復元
"""

import sys
from collections import deque

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

A, B = input().split()

# 例外処理
if A == B:
    print(0)
    print(A)
    print(B)
    exit()

N = INT()
M = len(A)
mapping = {}
rev = {}
S = []
for i in range(N):
    s = input()
    mapping[s] = i
    rev[i] = s
    S.append(s)
# 始点と終点を追加
mapping[A] = N
rev[N] = A
S.append(A)
mapping[B] = N + 1
rev[N+1] = B
S.append(B)

# 単語間を繋ぐグラフを構築する
nodes = [[] for i in range(N+2)]
for i, s in enumerate(S):
    for j, t in enumerate(S[i+1:], i+1):
        if s == t:
            continue
        cnt = 0
        for k in range(M):
            if s[k] == t[k]:
                cnt += 1
        # 1文字違いであれば辺を張る
        if cnt == M-1:
            a = mapping[s]
            b = mapping[t]
            nodes[a].append(b)
            nodes[b].append(a)

# 前の頂点を記録しながらBFS
que = deque([(N, 0)])
memo = [(INF, -1) for i in range(N+2)]
while que:
    u, cost = que.popleft()
    cost += 1
    for v in nodes[u]:
        if memo[v][0] == INF:
            memo[v] = (cost, u)
            que.append((v, cost))

# 到達ルートがない
if memo[N+1][0] == INF:
    print(-1)
    exit()

# s,t間の経路を取得
def get_route(s, t, res):
    prev = t
    StoT = [t]
    while prev != s:
        prev = res[prev][1]
        if prev == -1:
            return None
        StoT.append(prev)
    StoT = StoT[::-1]
    return StoT

# 経路復元
StoT = get_route(N, N+1, memo)
print(memo[N+1][0]-1)
for a in StoT:
    print(rev[a])
