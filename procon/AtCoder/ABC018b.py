# -*- coding: utf-8 -*-

S = input()
N = int(input())
for i in range(N):
    l, r = map(int, input().split())
    tmp = S[l-1:r]
    tmp = tmp[::-1]
    S = S[:l-1] + tmp + S[r:]
print(S)
