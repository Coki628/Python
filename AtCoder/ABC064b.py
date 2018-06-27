# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

aN.sort()

print(aN[N-1] - aN[0])