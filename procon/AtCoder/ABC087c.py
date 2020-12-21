# -*- coding: utf-8 -*-

N = int(input())
A1 = list(map(int, input().split()))
A2 = list(map(int, input().split()))

ans = 0
for i in range(N):
    tmp = 0
    tmp += sum(A1[0:i+1])
    tmp += sum(A2[i:N+1])
    ans = max(ans, tmp)

print(ans)