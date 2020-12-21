# -*- coding: utf-8 -*-

"""
・基本のナップザックでnumpy実装の練習、遷移を高速化。
・pythonTLEだったのが0.4秒AC。pypyより速くなった。
"""

import sys
from operator import itemgetter
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
INF = float('inf')
MOD = 10 ** 9 + 7

N, W = MAP()

vN = [0] * N
wN = [0] * N
VW = []
for i in range(N):
    v, w = MAP()
    vN[i], wN[i] = v, w
    VW.append((v, w))

# 応用ナップザック(実は1次元DPでいける)
if max(vN) <= 1000:
    import numpy as np 

    INF = 10 ** 18
    # 荷物を全部入れた場合のvalueを内ループに使う
    max_v = sum(vN)
    # 荷物のvalueを添え字にして重さを配列内に格納するDP
    dp = np.full(max_v+1, INF, dtype=np.int64)
    prev = np.full(max_v+1, INF, dtype=np.int64)
    dp[0] = prev[0] = 0

    for v, w in VW:
        if v <= max_v:
            # まとめて遷移
            dp[v:] = np.minimum(dp[v:], prev[:max_v+1-v] + w)
        prev = dp.copy()

    # 重さWの範囲内で価値が最大のもの
    ans = max([v for v, w in enumerate(dp) if w <= W])
    print(ans)

# 普通のナップザック(実はこれも1次元DPでいける)
elif max(wN) <= 1000:
    import numpy as np

    # 荷物の重さを添え字にしてvalueを配列内に格納するDP
    dp = np.zeros(W+1, dtype=np.int64)
    prev = np.zeros(W+1, dtype=np.int64)

    for v, w in VW:
        if w <= W:
            # まとめて遷移
            dp[w:] = np.maximum(dp[w:], prev[:W+1-w] + v)
        prev = dp.copy()
    print(dp[W])

# 半分全列挙
else:
    # 前半分についての組み合わせをビット全探索
    vN1 = vN[:N//2]
    wN1 = wN[:N//2]
    pairs = []
    for i in range(1<<len(vN1)):
        v = w = 0
        for j in range(len(vN1)):
            if i>>j & 1 == 1:
                v += vN1[j]
                w += wN1[j]
        # この時点で既に重すぎるペアはいらない
        if w <= W:
            pairs.append([w, v])

    # 重さ昇順、価値の降順でソート(優先する方を後にやる)
    pairs.sort(key=itemgetter(1), reverse=True)
    pairs.sort(key=itemgetter(0))
    # 利点のないペアを書き換え
    for i in range(1, len(pairs)):
        w1, v1 = pairs[i-1]
        w2, v2 = pairs[i]
        # 重さが重いのに価値が低い
        if v1 > v2:
            v2 = v1
        # 価値が同じなのに重さが重い
        if v1 == v2 and w1 < w2:
            w2 = w1
        pairs[i-1][0], pairs[i-1][1] = w1, v1
        pairs[i][0], pairs[i][1] = w2, v2

    # 後半分についての組み合わせをビット全探索
    vN2 = vN[N//2:]
    wN2 = wN[N//2:]
    ans = 0
    for i in range(1<<len(vN2)):
        v2 = w2 = 0
        for j in range(len(vN2)):
            if i>>j & 1 == 1:
                v2 += vN2[j]
                w2 += wN2[j]
        if w2 <= W:
            # ちょうどよく収まる重さのindexをに二分探索で取得
            # 先頭要素でソートしておけば、ペアのままbisectの引数にできる
            # 重さが同じものもなるべく右に寄せたいので、価値をINFにしておく
            v1 = pairs[bisect_right(pairs, [W-w2, float('inf')])-1][1]
            ans = max(v1+v2, ans)
    print(ans)
