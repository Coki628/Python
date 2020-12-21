# -*- coding: utf-8 -*-

N = int(input())

ans = 0
for i in range(N, -1, -1):
    if i % 2 != 0:
        # 約数の個数
        res = 0
        for j in range(1, i+1):
            if i % j == 0:
                res += 1
        if res == 8:
            ans += 1
print(ans)