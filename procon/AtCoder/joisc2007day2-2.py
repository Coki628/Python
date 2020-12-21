# -*- coding: utf-8 -*-

"""
参考：https://comprolog.netlify.com/post/joi/2008spring2-2/
・二分探索、XY独立、尺取法
・にぶたん内にぶたんで20万にlog2つはpypy1秒きついと思ったけど、尺取でいけるって記事見て、
　ああ確かに、それならlog1つだしpypy間に合うかも、と思って実装。
・無事、自力実装でpypyAC0.4秒。(pythonも一応試したけどTLE)
・とはいえ尺取はやっぱバグらせやすいなーと思った。
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

# Dの幅をmとした時、全区間をカバーするのに必要な装置の数がN以内かどうか
def check(m):
    # X,Y独立に尺取する
    l = r = 0
    res1 = 0
    while l < M:
        while r < M and X[r] - X[l] <= m:
            r += 1
        res1 += 1
        l = r
    l = r = 0
    res2 = 0
    while l < M:
        while r < M and Y[r] - Y[l] <= m:
            r += 1
        res2 += 1
        l = r
    # X,Y合わせた必要な数がN以内ならOK
    return res1 + res2 <= N

N, M = MAP()
X = [0] * M
Y = [0] * M
for i in range(M):
    x, y = MAP()
    X[i] = x
    Y[i] = y

# X,Y独立にして、ソートしておく
X.sort()
Y.sort()
# Dの幅で二分探索
res = bisearch_min(-1, 10**9+7, check)
print(res)
