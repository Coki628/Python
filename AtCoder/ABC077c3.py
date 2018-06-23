# -*- coding: utf-8 -*-

# 2分検索ライブラリを使う
import bisect

N = int(input())
A_list = list(map(int, input().split()))
B_list = list(map(int, input().split()))
C_list = list(map(int, input().split()))

# 昇順ソート
A_list.sort()
C_list.sort()

ans = 0
for i in range(N):
    # 同じ値があった時に、leftは先頭、rightは最後尾のindexを返す
    ab = bisect.bisect_left(A_list, B_list[i])
    bc = N - bisect.bisect_right(C_list, B_list[i])
    # 返されたindexから、候補となるABとBCの数が分かるので、合計していく
    ans += ab * bc
print(ans)