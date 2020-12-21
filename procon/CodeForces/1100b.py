# -*- coding: utf-8 -*-

N, M = map(int, input().split())
aM = list(map(int, input().split()))

level = [0] * (N+1)
cnt = 0
ans = ['0'] * M
for i in range(M):
    if level[aM[i]] == 0:
        cnt += 1
    level[aM[i]] += 1
    if cnt == N:
        ans[i] = '1'
        for j in range(1, N+1):
            level[j] -= 1
            if level[j] == 0:
                cnt -= 1
print(''.join(ans))
