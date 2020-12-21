# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc041
・蟻本演習2-5-1
・二部グラフ判定応用
・上書きできる設定は、逆順に辿るとみなせば、最初に塗った色に確定できる。
・↑こういう考察、ほんと出来たら嬉しいけど出来ない。。
・奇数長の閉路がある→二部グラフではない。
・奇数長の閉路があると、一気に出来る動作が増える。(今回なら、その先の色が自由に塗れる)
・計算量2000^2=400万でpythonTLE、pypyAC0.6秒。
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

N, M = MAP()
nodes = tuple([] for i in range(N))
for i in range(M):
    a, b, c = input().split()
    a = int(a) - 1
    b = int(b) - 1
    c = 0 if c == 'r' else 1
    nodes[a].append((b, c))
    nodes[b].append((a, c))

# 進める色の辺に進みながら、二部グラフ判定もやっていく
def check(u, startc):
    color = [-1] * N
    stack = [(u, -1, startc)]
    cnt = 0
    while stack:
        u, prev, c = stack.pop()
        if color[u] == -1:
            color[u] = c
        else:
            if color[u] == c:
                continue
            else:
                # 走査の最中に奇数長閉路を発見したら即OK
                return True
        for v, nxtc in nodes[u]:
            # 次の辺が塗りたい方の色なら進める
            if v != prev and nxtc == c:
                stack.append((v, u, 1-c))
                cnt += 1
    # 二部グラフだった時は、全ての辺を通れていればOK
    if cnt == M:
        return True
    else:
        return False

for c in range(2):
    for i in range(N):
        if check(i, c):
            Yes()
            exit()
No()
