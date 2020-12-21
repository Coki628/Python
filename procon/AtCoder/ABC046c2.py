# -*- coding: utf-8 -*-

N = int(input())

T,A = map(int, input().split())

for i in range(N-1):
    t,a = map(int, input().split())
    tmp1,tmp2 = 1,1
    # ループで足していくより、1回で出せる方法にする
    if t < T:
        # 切り上げはmath.ceilより演算子のみのが速い
        tmp1 = -(-T // t)
    if a < A:
        tmp2 = -(-A // a)
    # 多く増やす必要のある方に合わせて、同じ比で両方増やす
    T = t * max(tmp1,tmp2)
    A = a * max(tmp1,tmp2)

print(T + A)