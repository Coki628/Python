"""
・1つ前の深さの頂点数だけ分かってれば、どの頂点なのかは関係ないので、それを踏まえて簡潔に。
・popしてたところをiを進める感じに。これだと最初やろうとした実装方針にだいぶ近い。
・いやー、まじでこれは解けるべきだったな。。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N = INT()
    A = LIST()

    depth = 0
    parcnt = 1
    i = 1
    while i < N:
        nxt = 0
        for _ in range(parcnt):
            ch = -1
            while i < N and ch < A[i]:
                ch = A[i]
                i += 1
                nxt += 1
        parcnt = nxt
        depth += 1
    print(depth)
