# -*- coding: utf-8 -*-

import math

# 階乗しないで求める版
# 素数数え上げないで計算する版

x = int(input())

# 素数判定用関数
def is_prime_2(num):
    if num < 2:
        return False
    if num == 2 or num == 3 or num == 5:
        return True
    if num % 2 == 0 or num % 3 == 0 or num % 5 == 0:
        return False
    # 疑似素数(2でも3でも5でも割り切れない数字)で次々に割っていく
    prime = 7
    step = 4
    num_sqrt = math.sqrt(num)
    while prime <= num_sqrt:
        if num % prime == 0:
            return False
        prime += step
        step = 6 - step
    return True

# xまでの間にある素数を列挙
sosu_list = []
for i in range(x+1):
    if is_prime_2(i):
        sosu_list.append(i)

# 各素数毎に、1つずつ数え上げないで計算する
sosu_cnt = {}
for sosu in sosu_list:
    i = sosu
    while i <= x:
        if sosu in sosu_cnt:
            sosu_cnt[sosu] += x // i
        else:
            sosu_cnt[sosu] = x // i
        i *= sosu

# 素数カウントを使って、約数の個数を求める
ans = 1
for key, value in sosu_cnt.items():
    sosu_cnt[key] += 1
    ans = (ans * sosu_cnt[key]) % (10 ** 9 + 7)
print(ans)