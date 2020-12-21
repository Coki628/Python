# -*- coding: utf-8 -*-

"""
自力AC、想定解。
・素数判定
・累積和
"""

from math import sqrt

# 素数判定用関数
def is_prime2(num):
    if num < 2:
        return False
    if num == 2 or num == 3 or num == 5:
        return True
    if num % 2 == 0 or num % 3 == 0 or num % 5 == 0:
        return False
    # 疑似素数(2でも3でも割り切れない数字)で次々に割っていく
    prime = 7
    step = 4
    num_sqrt = sqrt(num)
    while prime <= num_sqrt:
        if num % prime == 0:
            return False
        prime += step
        step = 6 - step
    return True

# 同じ数2回素数判定やりたくないのでここに保存
prime = [is_prime2(i) for i in range(100001)]
# 2017に似た数
like2017 = [0] * 100001
for i in range(1, 100001, 2):
    if prime[i] and prime[(i+1)//2]:
        like2017[i] = 1
# 累積和
sm = [0] * 100001
for i in range(1, 100001):
    sm[i] = sm[i-1] + like2017[i]

Q = int(input())
for i in range(Q):
    l, r = map(int, input().split())
    # 累積和から各QはO(1)で出せる
    print(sm[r]-sm[l-1])
