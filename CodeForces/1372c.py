"""
・ここまで自力AC
・ギャグ。実は2回あれば何でもできる。0,1回で済むところだけ確認する。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

def RLE(data):
    from itertools import groupby

    return [(x, len(list(grp))) for x, grp in groupby(data)]

for _ in range(INT()):
    N = INT()
    A = LIST()

    B = [0] * N
    for i in range(N):
        if A[i] == i+1:
            B[i] = 1

    rle = RLE(B)
    cnt = 0
    for x, ln in rle:
        if x == 0:
            cnt += 1
    ans = min(cnt, 2)
    print(ans)
