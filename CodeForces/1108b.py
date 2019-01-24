# -*- coding: utf-8 -*-

from math import sqrt

def num_div_set2(N):
    s = {1, N}
    end = int(sqrt(N)) + 1
    for i in range(2, end+1):
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

N = int(input())
dN = sorted(map(int, input().split()))

l1 = sorted(num_div_set2(max(dN)))
for i in range(len(l1)):
    for j in range(N):
        if l1[i] == dN[j]:
            dN[j] = -1
            break
print(l1[-1], max(dN))
