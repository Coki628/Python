# -*- coding: utf-8 -*-

N,M = map(int, input().split())
X,Y = map(int, input().split())
aN = list(map(int, input().split()))
bM = list(map(int, input().split()))

i,j,cnt,time = 0,0,0,0
# 最後の便まで確認したら終了
while i < N and j < M:
    # AからBへ行く
    while i < N:
        # 時刻が出発時刻以内なら出発
        if time <= aN[i]:
            time = aN[i]
            i += 1
            time += X
            break
        # 間に合わないなら次の便へ
        else:
            i += 1
    # BからAへ行く
    while j < M:
        if time <= bM[j]:
            time = bM[j]
            j += 1
            time += Y
            # 往復できたらカウント
            cnt += 1
            break
        else:
            j += 1
print(cnt)