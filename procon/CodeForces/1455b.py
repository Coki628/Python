"""
・自力AC。でも結構悩んだ。
・等差数列、二分探索
・題意の操作を考えた時、まず全ての操作で+向きに動くとすると等差数列になり、
　そこから1つずつ-に向きを変えるとすると、その差分は先頭から-2,-3,-4..となっていく。
　これらをうまく組み合わせることで、等差数列の和,そこから-2,-3,-4...は全て構成できる。
　以上より、まず等差数列の和をN以上になるところまで伸ばして、
　その結果得られた等差数列の和から、それ-1以外は全てその回数で構成できて、-1だけはもう1回必要、となる。
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

def bisearch_min(mn, mx, func):

    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

for _ in range(INT()):
    N = INT()

    res = bisearch_min(0, INF, lambda m: (m+1)*m // 2 >= N)
    num = (res+1)*res // 2
    if num-1 == N:
        print(res+1)
    else:
        print(res)
