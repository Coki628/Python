# -*- coding: utf-8 -*-

N, M = map(int, input().split())
memo = [{'l': {}, 'r': {}} for i in range(N+1)]

for i in range(M):
    l, r, d = map(int, input().split())
    if d not in memo[l]['r'] or memo[l]['r'][d] == r:
        memo[l]['r'][d] = r
    else:
        print('No')
        exit()
    if d not in memo[r]['l'] or memo[r]['l'][d] == l:
        memo[r]['l'][d] = l
    else:
        print('No')
        exit()
print('Yes')
