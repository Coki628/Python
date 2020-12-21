# -*- coding: utf-8 -*-

N = int(input())

T,A = map(int, input().split())

for i in range(N-1):
    t,a = map(int, input().split())
    # 多く増やす必要のある方に合わせて、同じ比で両方増やす
    tmp = max(-(-T // t), -(-A // a))
    T = t * tmp
    A = a * tmp

print(T + A)