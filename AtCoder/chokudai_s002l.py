# -*- coding: utf-8 -*-

"""
・なんとか自力AC
・ABC038Dとほぼ同じ。
・縦の同値同士を降順にしておくことで、
　横でLIS取る時にそこは1つだけをカウントしてくれるようになるのでうまくいく。
・計算量が20万とlogで、pythonAC1.0秒、pypyAC1.1秒。
　pypyなんか遅いな、20万のソートのlogが重いのかな。
"""

import sys
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def LIS(A: list, equal=False) -> list:
    """ 最長増加部分列 """
    from operator import gt, ge
    from bisect import bisect_left, bisect_right

    # デフォルトは狭義のLIS(同値を含まない)
    compare = gt if not equal else ge
    bisect = bisect_left if not equal else bisect_right
    L = [A[0]]
    for a in A[1:]:
        if compare(a, L[-1]):
            # Lの末尾よりaが大きければ増加部分列を延長できる
            L.append(a)
        else:
            # そうでなければ、「aより小さい最大要素の次」をaにする
            # 該当位置は、二分探索で特定できる
            L[bisect(L, a)] = a
    return L

N = INT()
AB = []
for i in range(N):
    a, b = MAP()
    a, b = min(a, b), max(a, b)
    AB.append((a, b))

# 縦の昇順、横の降順に並べておく
AB.sort(key=itemgetter(1), reverse=1)
AB.sort(key=itemgetter(0))

# 横でLIS
_, B = zip(*AB)
ans = len(LIS(B))
print(ans)
