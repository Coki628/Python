# -*- coding: utf-8 -*-

"""
参考：http://ferin-tech.hatenablog.com/entry/2017/10/28/231616
・愚直に3重ループでシミュレーション
・計算量：O(N*max(tN[i]*2)*N)=100*400*100=400万 →これはpythonでも1秒くらいで通った。
"""

N = int(input())
tN = list(map(int, input().split()))
vN = list(map(int, input().split()))
# 終点の処理をしやすくするため追加
tN.append(0)
vN.append(0)

dist = 0
speed = 0
for i in range(N):
    keep = True
    plus = True
    # 小数刻みのfor文
    for j in [0.5*x for x in range(tN[i]*2)]:
        # 現在の最大スピードに達した
        if speed == vN[i]:
            plus = False
        rest = tN[i] - j
        # 今より先で、下げきれなくなる状況がないか確認
        for k in range(i+1, N+1):
            if vN[k] < vN[i] and rest <= speed - vN[k]:
                keep = False
                plus = False
            rest += tN[k]
        # フラグの状況に合わせて加速or維持or減速
        if plus:
            # ここでの数値の加減は進み方が0.5刻みなので注意する
            dist += speed/2
            speed += 0.5
            dist += 0.125
        elif keep:
            dist += speed/2
        else:
            speed -= 0.5
            dist += speed/2
            dist += 0.125
print(dist)
