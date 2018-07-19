# -*- coding: utf-8 -*-

N,T = map(int, input().split())
tN = list(map(int, input().split()))

ans = T
for i in range(1,N):

    if tN[i] - tN[i-1] < T:
        ans += tN[i] - tN[i-1]
    else:
        ans += T

print(ans)