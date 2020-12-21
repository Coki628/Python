# -*- coding: utf-8 -*-

"""
・蟻本演習2-5-1、自力AC！
・二部グラフ判定
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

N = INT()
nodes = tuple([] for i in range(N))
S = set()
for i in range(N):
    a = INT()
    a -= 1
    i, a = min(i, a), max(i, a)
    if (i, a) not in S:
        nodes[i].append(a)
        nodes[a].append(i)
        S.add((i, a))

# 二部グラフ判定
ans = []
color = [-1] * N
def check(u):
    stack = [(u, -1, 0)]
    cnt = [0] * 2
    while stack:
        u, prev, c = stack.pop()
        if color[u] == -1:
            color[u] = c
            cnt[c] += 1
        else:
            if color[u] == c:
                continue
            else:
                return False
        for v in nodes[u]:
            if v != prev:
                stack.append((v, u, 1-c))
    # 二部グラフの多い方の集合を答えに足す
    ans.append(max(cnt))
    return True

for i in range(N):
    if color[i] == -1:
        if not check(i):
            print(-1)
            exit()
print(sum(ans))
