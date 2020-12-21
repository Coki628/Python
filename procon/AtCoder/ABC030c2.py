# -*- coding: utf-8 -*-

# 結局人のを参考にして、現在位置を保持した版
# (でもこれ現在位置なくても大丈夫な気がするんだけどなぁ)

N,M = map(int, input().split())
X,Y = map(int, input().split())
aN = list(map(int, input().split()))
bM = list(map(int, input().split()))

i,j,cnt,time = 0,0,0,0
now = 'A'
# 最後の便まで確認したら終了
while i < N and j < M:
    # AからBへ行く
    while i < N and now == 'A':
        # 時刻が出発時刻以内なら出発
        if time <= aN[i]:
            time = aN[i]
            i += 1
            time += X
            now = 'B'
            break
        # 間に合わないなら次の便へ
        else:
            i += 1
    # BからAへ行く
    while j < M and now == 'B':
        if time <= bM[j]:
            time = bM[j]
            j += 1
            time += Y
            now = 'A'
            # 往復できたらカウント
            cnt += 1
            break
        else:
            j += 1
print(cnt)