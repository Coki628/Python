# -*- coding: utf-8 -*-

"""
・500点自力AC！
・グラフ系
・最初、問題の意味がよく掴めなかった。よく読んで、サンプル吟味して、理解。
・木の走査みたいに、適当な頂点から両側に伸ばせるだけ伸ばしたらいい感じな気がしたらそれでOKだった。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def rec(u, prev, li):
    li.append(u+1)
    visited[u] = 1
    for v in nodes[u]:
        if v != prev and visited[v] == -1:
            if rec(v, u, li):
                return True
    # 1回潜り切ったらそこで終了
    return True

N, M = MAP()
nodes = [[] for i in range(N)]
visited = [-1] * N
for i in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

ans = []
# とりあえず前へ
rec(0, -1, ans)
# 次は後ろ、逆向きに繋げるのでひっくり返しておく
ans = ans[1:][::-1]
rec(0, -1, ans)

print(len(ans))
print(*ans)
