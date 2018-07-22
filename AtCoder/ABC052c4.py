# -*- coding: utf-8 -*-

import math

# 階乗しないで求める版

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

# 素数のカウントは階乗前の値から個別に取って合算でも問題ない
sosu_cnt = {}
for i in range(1, x+1):
    for sosu in sosu_list:
        while i % sosu == 0:
            if sosu in sosu_cnt:
                sosu_cnt[sosu] += 1
            else:
                sosu_cnt[sosu] = 1
            i //= sosu

# 素数カウントを使って、約数の個数を求める
ans = 1
for key, value in sosu_cnt.items():
    sosu_cnt[key] += 1
    ans = (ans * sosu_cnt[key]) % (10 ** 9 + 7)
print(ans)