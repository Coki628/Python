"""
・自力AC
・グラフ、門松列
・元々想定解O(M^2)だったみたいだけど、O(N+M)で解けた。
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
EPS = 10 ** -10

N, M = MAP()
A = LIST()
nodes = [[] for i in range(N)]
for i in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(A[b])
    nodes[b].append(A[a])

# 門松列の真ん中を固定して考える
for i in range(N):
    S1 = set()
    S2 = set()
    for a in nodes[i]:
        if a < A[i]:
            S1.add(a)
        elif a > A[i]:
            S2.add(a)
    # 同じ長さでない自分より小さいor大きい頂点が2つ以上あれば門松列が成立する
    if len(S1) >= 2 or len(S2) >= 2:
        YES()
        exit()
NO()
