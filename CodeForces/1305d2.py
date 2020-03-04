# -*- coding: utf-8 -*-

"""
・自力ならず。ツイート参考でAC。
・インタラクティブ問題、グラフ、木の走査
・葉を順番にチェックして潰していく。
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
    def rec(u, prev):
        # 出ている辺の数(親とNG確定除く)を数える
        chdcnt = 0
        for v in nodes[u]:
            if v != prev and not NG[v]:
                chdcnt += 1
        # 辺が0なら葉とみなしてクエリ候補に入れる
        if chdcnt == 0:
            res.append(u)
            # 2つ集まったら再帰を終了させる
            if len(res) == 2:
                return True
            else:
                return False
        for v in nodes[u]:
            if v != prev and not NG[v]:
                if rec(v, u):
                    return True
        return False
    rec(src, -1)
    return res

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
    res = dfs(nodes, cur)
    if len(res) == 2:
        q = (res[0], res[1])
    elif len(res) == 1:
        q = (cur, res[0])
    else:
        break
    print('? {0} {1}'.format(q[0]+1, q[-1]+1))
    sys.stdout.flush()
    v = INT() - 1
    for u in q:
        if v == u:
            # 葉でありLCAだったものは根
            cur = u
            print('! {0}'.format(cur+1))
            exit()
        else:
            # 葉でありLCAでなかったものは根ではないので、次回以降の探索候補から外す
            NG[u] = 1
            ngcnt += 1
    cur = v
print('! {0}'.format(cur+1))
