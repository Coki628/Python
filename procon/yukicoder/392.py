"""
・自力AC
・2分木の構造、BFSに帰着
・想定解と違ったけど、採番の仕方がBFSっぽかったので根からBFSした。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def bfs():
    from collections import deque

    N = 4095
    que = deque([''])
    res = [''] * N
    num = 0
    while que:
        s = que.popleft()
        res[num] = s
        num += 1
        if num == N:
            break
        que.append(s+'L')
        que.append(s+'R')
    return res
ans = bfs()

M = INT()
for _ in range(M):
    a = INT()
    print(ans[a])
