# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/joi/2017/2018-ho/2018-ho-t2-review.pdf
　　　https://www.ioi-jp.org/joi/2017/2018-ho/2018-ho-t2-sample.cpp
・自力ならず。。
・式変形
・部分点のO(N^2)解法までは公式と同じ方針が出てたけど、そこまで。
・S - (max(A) - min(A)) について、Aでソート済の列を考えると、
　ある区間[i, j]について、Sは累積和にすると、区間和で Sj - Si-1 とできる。
　Aはそのまま最小最大が左端と右端なので Aj - Ai になる。
　まとめると、 (Sj - Si-1) - (Aj - Ai) これを式変形して (Sj - Aj) - (Si-1 - Ai)
　ここまで来ると、i, jが独立になるので、左端となるiの最小をすぐに見つけられる。
・実際の動きはソース内コメントも参照。
・計算量は50万のNにソートのlog、これが1秒だとpypyでも通らなかった。。
・C++でAC0.2秒。さすが。
"""

import sys
from operator import itemgetter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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
AB = []
for _ in range(N):
    a, b = MAP()
    AB.append((a, b))
AB.sort(key=itemgetter(0))

mn = acc = ans = 0
for a, b in AB:
    # min(min(Si-1 - Ai), Sj-1 - Aj) ：左端の最小値を更新
    mn = min(mn, acc - a)
    # Sj = Sj-1 + Bj ：Bの累積和を更新
    acc += b
    # (Sj - Aj) - min(Si-1 - Ai) ：答えの最大値を更新
    ans = max(ans, (acc - a) - mn)
print(ans)
