# -*- coding: utf-8 -*-

N = int(input())
S = input()

ans = float("inf")
for i in range(N):
    cnt = 0
    cnt += S[:i].count("W")
    cnt += S[i+1:].count("E")
    ans = min(ans, cnt)
print(ans)