# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc032
　　　http://mayokoex.hatenablog.com/entry/2016/01/09/230000
・DP2つ
・半分全列挙
・応用DPが重いっぽくてpythonだとTLEでpypyでAC、0.8秒くらい。
・応用DPの計算量：O(N*sum(V))=200*200*1000=4000万だからきついわけだ。
"""

from operator import itemgetter
from bisect import bisect_right

N, W = map(int, input().split())

vN = [0] * N
wN = [0] * N
for i in range(N):
    vN[i], wN[i] = map(int, input().split())

# 応用ナップザック(実は1次元DPでいける)
if max(vN) <= 1000:
    # 荷物を全部入れた場合のvalueを内ループに使う
    max_v = sum(vN)
    # 荷物のvalueを添え字にして重さを配列内に格納するDP
    dp = [float('inf')] * (max_v+1)
    dp[0] = 0
    for i in range(N):
        for j in range(max_v, -1, -1):
            if j >= vN[i]:
                dp[j] = min(dp[j-vN[i]] + wN[i], dp[j])
    ans = 0
    # 重さWの範囲内で価値が最大のもの
    for i in range(max_v+1):
        if dp[i] <= W:
            ans = max(ans, i)
    print(ans)

# 普通のナップザック(実はこれも1次元DPでいける)
elif max(wN) <= 1000:
    # 荷物の重さを添え字にしてvalueを配列内に格納するDP
    dp = [0] * (W+1)
    for i in range(N):
        for j in range(W, -1, -1):
            if j >= wN[i]:
                dp[j] = max(dp[j-wN[i]] + vN[i], dp[j])
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
            pairs.append([v, w])

    # 重さ昇順、価値の降順でソート(優先する方を後にやる)
    pairs.sort(key=itemgetter(0), reverse=True)
    pairs.sort(key=itemgetter(1))
    # 利点のないペアを書き換え
    for i in range(1, len(pairs)):
        v1, w1 = pairs[i-1]
        v2, w2 = pairs[i]
        # 重さが重いのに価値が低い
        if v1 > v2:
            v2 = v1
        # 価値が同じなのに重さが重い
        if v1 == v2 and w1 < w2:
            w2 = w1
        pairs[i-1][0], pairs[i-1][1] = v1, w1
        pairs[i][0], pairs[i][1] = v2, w2

    # この後の二分探索のキー用に重さだけ分離
    keys = [w for v, w in pairs]

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
            # 欲しいのは1つ前の値なのでright-1する
            # 常に(0, 0)があるので、先頭に入って添字が-1になることはない
            v1 = pairs[bisect_right(keys, W-w2)-1][0]
            ans = max(v1+v2, ans)
    print(ans)
