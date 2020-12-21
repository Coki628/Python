# -*- coding: utf-8 -*-

N = int(input())
A = int(input())

# 500円のみで払える
if N / 500 == 0:
    print("Yes")
# 1円を使う
else:
    # 500の剰余を出して、手持ちの1円で払いきれるか判断
    mod = N % 500
    if A >= mod:
        print("Yes")
    else:
        print("No")