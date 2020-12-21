# -*- coding: utf-8 -*-

"""
・結構すんなり自力AC！
・でも貪欲詰めこわい。いけるかも、と考えてはいても、考慮漏れがない確証は持てないから。
"""

import sys
from itertools import accumulate

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

N = INT()
A = LIST(N)

# lim := 今使っていい最小値(これを下回ると0になる人が出る)
lim = 1
ans = 0
for a in A:
    d, m = divmod(a, lim)
    # この人からは取れない
    if d == 0 or d == 1 and m == 0:
        # この人の値は変化しないので、次からのリミットはこの人+1
        lim = max(lim, a+1)
    else:
        # 0にしちゃいけないので、割り切れるなら-1引いた値で追加
        ans += d - (m == 0)
        # 途中うまく調整してこの人は1にできるはずなので、次からのリミットは2
        lim = max(lim, 2)
print(ans)
