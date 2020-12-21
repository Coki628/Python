# -*- coding: utf-8 -*-
 
N = int(input())
 
T,A = map(int, input().split())
 
for i in range(N-1):
    t,a = map(int, input().split())
    tmp1,tmp2 = t,a
    # 比はそのままで、あり得る投票数まで値を増やす
    while t < T or a < A:
        t += tmp1
        a += tmp2
    T = t
    A = a
 
print(T + A)