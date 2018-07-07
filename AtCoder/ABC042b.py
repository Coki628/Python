# -*- coding: utf-8 -*-

N, L = map(int, input().split())
str_list = [input() for i in range(N)]

str_list.sort()

ans = ""
for i in range(N):
    ans += str_list[i]

print(ans)