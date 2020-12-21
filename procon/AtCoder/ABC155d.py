# -*- coding: utf-8 -*-

"""
・自力はならず。時間なかった。終了後、だいたい自力AC。
・条件が複雑な二分探索、二分探索内で二分探索
・方針は悪くなかった。基本は「億マス計算」ベースなんだけど、
　0と負数があるので、適宜場合分けする。
・具体的には、負数、0、正数で分けて、それぞれを組み合わせたパターンを個別にやる。
・6パターンある。詳細は自分ノート参照。
・計算量きつくて、20万のNにlog2つかかる。多分億超え。pypyでもTLE。
・内側をにぶたんじゃなくて尺取りにすればlog1つ外れてACできるみたいだけど、
　だるいからC++でAC0.7秒。負数除算の仕様の違いを学んだ。
"""

import sys
from bisect import bisect_right

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

def bisearch_min(mn, mx, func):
    """ 条件を満たす最小値を見つける二分探索 """

    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            # 下を探しに行く
            ok = mid
        else:
            # 上を探しに行く
            ng = mid
    return ok

# x以下の数がK個以上かどうか
def check(x):
    # 各aについて、条件を満たす数を数える
    ans = 0
    # + * +
    for i, a in enumerate(A3):
        # 式変形： a * b <= x → b <= floor(x, a)
        # 重複カウントを避けるため、indexを調整
        ans += max(bisect_right(A3, x // a) - (i+1), 0)
    # - * - 
    for i, a in enumerate(A1b):
        ans += max(bisect_right(A1b, x // a) - (i+1), 0)
    # - * +
    for i, a in enumerate(A3):
        ans += bisect_right(A1, x // a)
    # 0 * -, 0 * +, 0 * 0
    cnt0 = N2*N1 + N2*N3 + N2*(N2-1)//2
    if x >= 0:
        ans += cnt0
    return ans >= K

N, K = MAP()
A = LIST()

A.sort()
# -, 0, + をそれぞれ別配列に分ける
A1, A1b, A2, A3 = [], [], [], []
for a in A:
    if a < 0:
        A1.append(a)
        A1b.append(-a)
    elif a == 0:
        A2.append(a)
    else:
        A3.append(a)
A1b = A1b[::-1]
N1 = len(A1)
N2 = len(A2)
N3 = len(A3)

print(bisearch_min(-INF, INF, check))
