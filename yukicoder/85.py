"""
・結構すんなり自力AC
・場合分け
・縦横どっちかが1の場合は例外処理。それ意外はマス数が偶数なら戻ってこれるし奇数なら無理。
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

H, W, C = MAP()

if H == 1:
    if W == 2:
        YES()
    else:
        NO()
    exit()
if W == 1:
    if H == 2:
        YES()
    else:
        NO()
    exit()

if H * W % 2 == 0:
    YES()
else:
    NO()
