# -*- coding: utf-8 -*-

from itertools import product

N, K = map(int, input().split())
tNK = [] * N
for i in range(N):
    tNK.append(list(map(int, input().split())))

# 引数リストのアンパック(*args => arg1, arg2みたいになる)
for nums in list(product(*tNK)):
    ans = 0
    for num in nums:
        # 全部XORする
        ans ^= num
    # 0が1回でも出ればバグ
    if ans == 0:
        print('Found')
        exit()
# なければOK
print('Nothing')
