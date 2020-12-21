# -*- coding: utf-8 -*-

N = int(input())
S = [''] * N
P = [0] * N
for i in range(N):
    S[i], p = map(str, input().split())
    P[i] = int(p)

name = ''
ppl = 0
for i in range(N):
    if P[i] >= ppl:
        ppl = P[i]
        name = S[i]

if ppl > sum(P) / 2:
    print(name)
else:
    print('atcoder')