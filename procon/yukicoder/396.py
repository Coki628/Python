"""
・さくっと自力AC
・想定解は周期2倍にしてたから、ちょっと違う解法。
・何番目の周期にいるかが(0-indexedで)奇数なら反転、ってやった。
　これで向きが同じになって周期内での位置が一致してるか判定できる。
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

N, M = MAP()
x, y = MAP()
x -= 1; y -= 1

dx, mx = divmod(x, M)
dy, my = divmod(y, M)
# 奇数番目の周期にいたら反転させる
if dx % 2 == 1:
    mx = M - mx - 1
if dy % 2 == 1:
    my = M - my - 1

if mx == my:
    YES()
else:
    NO()
