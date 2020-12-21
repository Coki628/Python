"""
・自力AC！
・整数BFS、逆操作
・よくある感じの。ある数から0を目指す。今回は操作回数いらなくて到達の可否だけなのでdictはsetにした。
・明らかにlogNくらいで届きそうな操作なのに、なぜかMLE,TLEが取れなくてなんでかなー、と思ったら、
　片方が0になった時、無限に-1しまくってたね。片方が0になったらあとはひたすら引けばいいんだから
　到達可で確定だね。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def bfs(src):
    """ BFS(整数) """
    from collections import deque

    que = deque([src])
    dist = set()
    dist.add(src)
    while que:
        x, y = que.popleft()
        if x == 0 or y == 0:
            return True
        if x % 2 == 0 and y > 0:
            nx = x // 2
            ny = y - 1
            if (nx, ny) not in dist:
                dist.add((nx, ny))
                que.append((nx, ny))
        if x > 0 and y % 2 == 0:
            nx = x - 1
            ny = y // 2
            if (nx, ny) not in dist:
                dist.add((nx, ny))
                que.append((nx, ny))
    return False

a, b = MAP()

res = bfs((a, b))
if res:
    Yes()
else:
    No()
