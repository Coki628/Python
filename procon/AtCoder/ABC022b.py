# -*- coding: utf-8 -*-

N = int(input())
aN = [int(input()) for i in range(N)]

# 訪問済の花の種類を保持
done = [False] * (10 ** 5)
ans = 0
for i in range(N):
    if done[aN[i]-1]:
        ans += 1
    done[aN[i]-1] = True
print(ans)
