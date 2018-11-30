# -*- coding: utf-8 -*-

import math

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

N = int(input())
aN = list(map(int, input().split()))

cnt = 0
for i in range(N):
    if is_prime_2(aN[i]):
        cnt += 1
print(cnt)
