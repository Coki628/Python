# -*- coding: utf-8 -*-

"""
色々整理版
"""

N = int(input())
aN = list(map(int, input().split()))

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
            # 同じ帽子の人数と人数カウントの一致を確認
            if aN[j][0] != cnt:
                print('Impossible')
                exit()
            aN[j].append(b)
        b += 1
        cnt = 0

# 最後まで帽子の種類が足されたかどうか
if len(aN[-1]) == 3: 
    # ソートしたやつ元の順番に戻す
    aN.sort(key=lambda x: x[1])
    print('Possible')
    for i in range(N):
        # こどふぉは最後の改行考慮しなくても通る
        print(aN[i][2], end=' ')
else:
    print('Impossible')
