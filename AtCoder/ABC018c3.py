# -*- coding: utf-8 -*-

"""
解説を参考に作成
＋ちょっと人の見て条件とか整理した版
"""

R,C,K = map(int, input().split())
field = []
for i in range(R):
    field.append(list(input()))

# 上下にどこまでいけるかを記録する
UD = [[float('inf')] * C for j in range(R)]
# 縦方向に先に動くループ
for i in range(C):
    cnt = 0
    # 上から下に見る
    for j in range(R):
        if field[j][i] == 'o':
            cnt += 1
        else:
            cnt = 0
        UD[j][i] = cnt
    cnt = 0
    # 下から上に見る
    for j in range(R-1, -1, -1):
        if field[j][i] == 'o':
            cnt += 1
        else:
            cnt = 0
        UD[j][i] = min(cnt, UD[j][i])

ans = 0
for i in range(K-1, R-K+1):
    for j in range(K-1, C-K+1):      
        valid = True
        # 左右方向の確認
        for k in range(1, K+1):
            if UD[i][j-K+k] < k or UD[i][j+K-k] < k:
                valid = False
        # 全て条件を満たせばOK
        if valid:
            ans += 1
print(ans)
