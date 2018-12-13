# -*- coding: utf-8 -*-

N = int(input())
aN = [int(input()) for i in range(N)]

aN.sort(reverse=True)
aN[0] //= 2
print(sum(aN))
