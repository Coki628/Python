# -*- coding: utf-8 -*-

K, S = map(int, input().split())

# 絶対TLE…
ans = 0
for x in range(K+1):
    for y in range(K+1):
        for z in range(K+1):
            if x + y + z == S:
                ans += 1
print(ans)