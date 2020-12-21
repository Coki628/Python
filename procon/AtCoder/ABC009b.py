# -*- coding: utf-8 -*-

N = int(input())
aN = [int(input()) for i in range(N)]

aN = sorted(set(aN), reverse=True)
print(aN[1])
