# -*- coding: utf-8 -*-

"""
・自力WA
・隣接する3点で調べて、根の方向に近づいていく。
・これでも大丈夫だと思うんだけど、なんかWAでダメだった。。
"""

import sys
from collections import Counter

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def dfs(nodes, src):
    res = []
    is_tmnl = True
    def rec(u, prev):
        global is_tmnl
        if prev != -1 and not NG[u]:
            res.append(u)
            if len(res) == 2:
                return True
        for v in nodes[u]:
            if v != prev:
                if rec(v, u):
                    return True
                if res and prev == -1:
                    is_tmnl = False
        return False
    rec(src, -1)
    return res, is_tmnl

N = INT()
nodes = [[] for i in range(N)]
for i in range(N-1):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

NG = [0] * N
ngcnt = 0
cur = 0
while ngcnt+1 < N:
    res, is_tmnl = dfs(nodes, cur)
    if len(res) == 2:
        if is_tmnl:
            q = (cur, res[0], res[1])
        else:
            q = (res[0], cur, res[1])
    elif len(res) == 1:
        q = (cur, res[0])
    else:
        break
    print('? {0} {1}'.format(q[0]+1, q[-1]+1))
    sys.stdout.flush()
    v = INT() - 1
    for u in q:
        if v != u:
            NG[u] = 1
            ngcnt += 1
    cur = v
print('! {0}'.format(cur+1))
