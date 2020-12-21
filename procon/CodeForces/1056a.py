# -*- coding: utf-8 -*-

N = int(input())
for i in range(N):
    lines = list(map(str, input().split()))
    lines = lines[1:]
    if i == 0:
        res = set(lines)
    else:
        res &= set(lines)
print(' '.join(res))
