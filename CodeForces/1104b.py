# -*- coding: utf-8 -*-

S = list(input())

i = 1
cnt = 0
while i < len(S) and len(S) >= 2:
    if S[i] == S[i-1]:
        S.pop(i-1)
        S.pop(i-1)
        i -= 2
        cnt += 1
    i += 1

if cnt % 2 == 0:
    print('No')
else:
    print('Yes')
