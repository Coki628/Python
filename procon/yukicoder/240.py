"""
・さくっと自力AC
・絶対頭いい数学O(1)解あるんだろうなって思いながらBFSナイトのために8方向書いた。
・そしたら意外にもBFS想定解だった。
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

def bfs(src):
    from collections import deque

    que = deque([src])
    dist = {}
    dist[src] = 0
    while que:
        x, y = que.popleft()
        if dist[(x, y)] >= 3:
            continue
        if (x-2, y-1) not in dist:
            dist[(x-2, y-1)] = dist[(x, y)] + 1
            que.append((x-2, y-1))
        if (x-2, y+1) not in dist:
            dist[(x-2, y+1)] = dist[(x, y)] + 1
            que.append((x-2, y+1))
        if (x-1, y-2) not in dist:
            dist[(x-1, y-2)] = dist[(x, y)] + 1
            que.append((x-1, y-2))
        if (x-1, y+2) not in dist:
            dist[(x-1, y+2)] = dist[(x, y)] + 1
            que.append((x-1, y+2))
        if (x+1, y-2) not in dist:
            dist[(x+1, y-2)] = dist[(x, y)] + 1
            que.append((x+1, y-2))
        if (x+1, y+2) not in dist:
            dist[(x+1, y+2)] = dist[(x, y)] + 1
            que.append((x+1, y+2))
        if (x+2, y-1) not in dist:
            dist[(x+2, y-1)] = dist[(x, y)] + 1
            que.append((x+2, y-1))
        if (x+2, y+1) not in dist:
            dist[(x+2, y+1)] = dist[(x, y)] + 1
            que.append((x+2, y+1))
    return dist

x, y = MAP()

res = bfs((0, 0))
if (x, y) in res:
    YES()
else:
    NO()
