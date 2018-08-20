# -*- coding: utf-8 -*-

N,A = map(int, input().split())
xN = list(map(int, input().split()))

ans = 0
for i in range(1,1<<N):
    sum1 = 0
    cnt = 0
    for j in range(N):
        if i & 1<<j:
            sum1 += xN[j]
            cnt += 1
    if sum1 / cnt == A:
        ans += 1
        
print(ans)