# -*- coding: utf-8 -*-

"""
・自力AC！
・後ろから前計算、貪欲シミュレーション
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

N = INT()
S = input()

# 各位置での次回出現位置を事前に取っておく
nxt = [INF] * N
# RGBの現在地点での次回出現位置
tmp = {'R': INF, 'G': INF, 'B': INF}
for i in range(N-1, -1, -1):
    s = S[i]
    nxt[i] = tmp[s]
    tmp[s] = i

que = deque()
for i, s in enumerate(S):
    # RGBの次回位置を更新
    tmp[s] = nxt[i]
    if len(que) == 0:
        que.append(s)
    elif len(que) == 1:
        t = que[0]
        if s == t:
            que.pop()
        else:
            que.append(s)
    else:
        t = que[0]
        u = que[-1]
        if s == t:
            que.popleft()
        elif s == u:
            que.pop()
        else:
            # 両端どちらでもないのが来たら、次回位置の近い方を残す
            if tmp[t] < tmp[u]:
                que.append(s)
            else:
                que.appendleft(s)
print(len(que))
