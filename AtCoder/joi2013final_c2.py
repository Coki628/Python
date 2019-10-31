# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-2
・尺取法、二分探索
・4点動かす尺取法。これも前ににぶたんで通したやつ。
・これの尺取の解法について、公式解説とか読んでもいまいち理解が進まなくて、
　諦め半分で自分で思ったように4点決めて組んだらうまくいった！ｗ
・4点もあるとややこしいだろうと思ったけど、意外と素直に前からシミュレーションしてく感じ。
・初回だけ各点の位置が決まってないために例外処理する部分があるから注意。
・さすがにlog乗らない分、にぶたんより速くなって、pypyTLE → pypyAC0.5秒。
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
INF = float('inf')
MOD = 10 ** 9 + 7

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索 """

    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

N = INT()
A = LIST(N)

# 全部始点にできるように、2周分作っておく
A2 = A + A
total = sum(A)
def check(m):
    # 尺取法(4点動かす)
    a = b = c = d = 0
    cnt1 = cnt2 = cnt3 = 0
    while a < N:
        # 1つ目のピースの終端を決める
        while b < N*2 and cnt1 < m:
            cnt1 += A2[b]
            # 最初以外は次のピースから引く
            if a != 0:
                cnt2 -= A2[b]
            b += 1
        # 最初だけは次のピースの位置合わせをする
        if a == 0:
            c = b
        # 2つ目のピースの終端を決める
        while c < N*2 and cnt2 < m:
            cnt2 += A2[c]
            if a != 0:
                cnt3 -= A2[c]
            c += 1
        if a == 0:
            d = c
        # 3つ目のピースの終端を決める
        while d < N*2 and cnt3 < m:
            cnt3 += A2[d]
            d += 1
        # 3つともm以上あって、合計の長さがN以内ならOK
        if cnt1 >= m and cnt2 >= m and cnt3 >= m and d - a <= N:
            return True
        cnt1 -= A2[a]
        a += 1
    # どの位置から始めてもダメならNG
    return False

res = bisearch_max(0, total//2, check)
print(res)
