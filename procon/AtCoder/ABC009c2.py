# -*- coding: utf-8 -*-

import copy

N, K = map(int, input().split())
S = list(input())

changed = [False] * N
init = copy.deepcopy(S)

for i in range(N):
    target = i
    for j in range(i+1, N):
        if S[j] < S[target]:
            target = j
    if target != i:
        changed[i] = True
        changed[target] = True
        if changed.count(True) > K:
            print(''.join(S))
            exit()
        tmp = S[i]
        S[i] = S[target]
        S[target] = tmp
        if S[i] == init[i]:
            changed[i] = False
        if S[target] == init[target]:
            changed[target] = False
print(''.join(S))
