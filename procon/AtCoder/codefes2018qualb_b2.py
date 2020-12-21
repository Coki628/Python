# -*- coding: utf-8 -*-

N,X = map(int, input().split())

ans = 0
aN,bN = [0]*N, [0]*N
for i in range(N):
    aN[i], bN[i] = map(int, input().split())
    ans += aN[i] * bN[i]

# 実は最後に追加分足すだけでいい
ans += X * max(bN)

print(ans)