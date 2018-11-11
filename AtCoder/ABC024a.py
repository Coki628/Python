# -*- coding: utf-8 -*-

A, B, C, K = map(int, input().split())
S, T = map(int, input().split())

total = S * A + T * B
if S + T >= K:
   total -= (S+T) * C
print(total)