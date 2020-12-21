# -*- coding: utf-8 -*-

n = int(input())
Sn = [input() for i in range(n)]

ans = ""
for i in range(97, 97+26):
    c = chr(i)
    mn = float("inf")
    for j in range(n):
        mn = min(mn, Sn[j].count(c))
    for j in range(mn):
        ans += c
print(ans)