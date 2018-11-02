# -*- coding: utf-8 -*-

# 人の参考にちょっと整理した版

N = int(input())
aN = list(map(int, input().split()))

ans = float('-inf')
# takaのターン
for i in range(N):
    # 毎回使い捨ての1次配列なのでここで宣言
    taka,aoki = [],[]
     # aokiのターン
    for j in range(N):
        # 同じ場所にはできない
        if i == j:
            continue
        # 特定の区間
        sliced = aN[min(i,j):max(i,j)+1]
        # 偶数index(奇数番目)をtaka
        taka.append(sum(sliced[::2]))
        # 奇数index(偶数番目)をaoki
        aoki.append(sum(sliced[1::2]))
    # aokiが最大(同値は一番左)の時のtaka
    ans = max(taka[aoki.index(max(aoki))], ans)
print(ans)