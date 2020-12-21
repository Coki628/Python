"""
・さくっと自力AC、条件漏れてて1ペナしたけど。。
・場合分け
・将棋の王が動くあの斜めありの距離はチェビシェフ距離って言うらしい。
・邪魔な駒は1つだけなので、条件整えて場合分けする。
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

x1, y1 = MAP()
x2, y2 = MAP()

dist = max(x1, y1)
if abs(x1) == abs(y1) and abs(x2) == abs(y2):
    if 0 < x2 < x1 and 0 < y2 < y1 \
            or x1 < x2 < 0 and y1 < y2 < 0:
        dist += 1
print(dist)
