"""
・自力AC！
・グラフ応用、誤差対策
・結構悩んだ。最小全域木作る時みたいに、距離短い辺から順に処理していく。
・1000頂点、50万辺くらいで、ソートきついかなって思ったらやっぱきつくて普通にやったらTLE。
　itemgetterでキー1つにしたら間に合った。
・でもWAが1つ。これは誤差。hypotで素直に距離取ってたのを、2乗のままにして整数でやる。
　これでAC。pypyで1.33秒。やっぱ数がでかいとソートのlog重いな。。
"""

import sys
from operator import itemgetter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
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

N = INT()
XY = [()] * N
for i in range(N):
    x, y = MAP()
    XY[i] = (x, y)

edges = []
for i, (x, y) in enumerate(XY):
    for j, (x2, y2) in enumerate(XY[i+1:], i+1):
        dist = (x-x2)**2 + (y-y2)**2
        edges.append((dist, i, j))

edges.sort(key=itemgetter(0))
removed = [0] * N
ans = 0
for dist, a, b in edges:
    if removed[a] or removed[b]:
        continue
    if a == 0:
        ans += 1
        removed[b] = 1
    elif b == 0:
        ans += 1
        removed[a] = 1
    else:
        removed[a] = 1
        removed[b] = 1
print(ans)
