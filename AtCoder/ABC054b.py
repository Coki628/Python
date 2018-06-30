# -*- coding: utf-8 -*-

N, M = map(int, input().split())
A = [list(input()) for i in range(N)]
B = [list(input()) for i in range(M)]

# AとBの大きさに応じたループ回数
for i in range(N - M + 1):
    for j in range(N - M + 1):
        # Mの行数で比較のループ
        for k in range(M):
            # 各行についてAとBの一致を確認
            if "".join(A[i+k])[j:j+M] != "".join(B[k]):
                break
            # 全ての行について一致したらYes
            if k == M - 1:
                print("Yes")
                exit()
# 条件に合った箇所がなければNo
print("No")