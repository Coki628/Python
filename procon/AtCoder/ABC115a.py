# -*- coding: utf-8 -*-

D = int(input())

cnt = 25 - D
S = ['Christmas']
for i in range(cnt):
    S.append('Eve')
print(*S)
