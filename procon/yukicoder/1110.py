"""
・自力AC
・ソートして、一時領域にキューっぽく詰めて、出せるとこまで出して数えるって感じ。
・想定解はにぶたんか尺取だったけど、まあ尺取に近いんかなー。
"""

import sys
from collections import deque

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N, D = MAP()
A = [(a, i) for i, a in enumerate(LIST(N))]

tmp = deque()
cur = 0
ans = [0] * N
A.sort()
for a, i in A:
    while tmp and tmp[0] <= a-D:
        tmp.popleft()
        cur += 1
    tmp.append(a)
    ans[i] = cur

for a in ans:
    print(a)
