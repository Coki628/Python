# -*- coding: utf-8 -*-

import copy

N = int(input())
tests = list(map(int, input().split()))
M = int(input())

for i in range(M):
    tmp = copy.deepcopy(tests)
    p, x = map(int, input().split())
    tmp[p-1] = x
    print(sum(tmp))