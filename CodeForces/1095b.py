# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

aN.sort()
print(min(aN[-1] - aN[1], aN[-2] - aN[0]))
