"""
・うまく分類してうまくソートする。詳細はソース内コメント。
"""

import sys
from collections import Counter
from operator import itemgetter

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

for _ in range(INT()):
    N, K = MAP()
    A = LIST()

    # Kの倍数にするのにあといくつ要るかでグループにする
    A = [K-a%K if a%K else 0 for a in A]
    C = Counter(A)
    # 既にKの倍数のやつは要らない
    if 0 in C:
        del C[0]
    # 数の多い方、数値の大きい方、の優先度でソート
    C = list(C.items())
    C.sort(reverse=1, key=itemgetter(0))
    C.sort(reverse=1, key=itemgetter(1))
    if C:
        # 一番時間かかるやつまでの操作回数
        ans = (C[0][1]-1) * K + C[0][0] + 1
        print(ans)
    else:
        print(0)
