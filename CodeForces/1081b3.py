# -*- coding: utf-8 -*-

"""
ギリギリ提出間に合わなかった版
"""

N = int(input())
aN = list(map(int, input().split()))

# 全員同じの時、別処理
if aN.count(0):
    if aN.count(0) == N:
        print('Possible')
        print(*([1] * N))
    else:
        print('Impossible')
    exit()

for i in range(N):
    # 同じ帽子の人数, 元の順番, 帽子の種類(後でappend)
    aN[i] = [N - aN[i], i+1]
aN.sort()

# 帽子の種類
b = 1
# 同じ帽子の人数カウント
cnt = 0
for i in range(N):
    cnt += 1
    if aN[i][0] == cnt:
        for j in range(i+1-cnt, i+1):
            aN[j].append(b)
        b += 1
        cnt = 0

if len(aN[-1]) == 3: 
    # ソートしたやつ元の順番に戻す
    aN.sort(key=lambda x: x[1])
    print('Possible')
    for i in range(N):
        if i != N-1:
            print(aN[i][2], end=' ')
        else:
            print(aN[i][2])
else:
    print('Impossible')
