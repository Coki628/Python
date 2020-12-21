# -*- coding: utf-8 -*-

N,S,T = map(int, input().split())
aN = [int(input()) for i in range(N)]

w,ans = 0,0
for i in range(N):
    w += aN[i]
    if S <= w <= T:
        ans += 1
print(ans)
