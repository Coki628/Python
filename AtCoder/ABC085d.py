# -*- coding: utf-8 -*-

"""
自力AC。公式解説の想定解と考え方も合ってるし、いい感じ！
"""

from bisect import bisect_left

N, H = map(int, input().split())
A, B = [0]*N, [0]*N
for i in range(N):
    A[i], B[i] = map(int, input().split())

a_max = max(A)
B.sort()
# 一番強い振る刀より弱い投げる刀はいらない
B2 = list(reversed(B[bisect_left(B, a_max):]))

cnt = 0
# 投げるだけで倒せるか
if H <= sum(B2):
    # 強い方から順番に投げる
    for i in range(len(B2)):
        cnt += 1
        H -= B2[i]
        if H <= 0:
            print(cnt)
            exit()
else:
    # 全部投げた結果
    H -= sum(B2)
    cnt += len(B2)
    # 倒すまでに振る回数
    cnt += -(-H // a_max)
    # ※ルール上、厳密には振るだけ振ってから全部投げるんだけど、
    # 　回数はどっちにしても変わらない。
    print(cnt)
