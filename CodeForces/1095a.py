# -*- coding: utf-8 -*-

N = int(input())
T = input()

i = 1
ans = ''
while T != '':
    ans += T[0]
    T = T[i:]
    i += 1
print(ans)
