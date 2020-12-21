# -*- coding: utf-8 -*-

n = int(input())

ans = n
for i in range(1,n+1):
    j = 1
    while i * j <= n:
        # 全体 - 四角形 + 長辺と短辺の差分
        ans = min(n - i*j + abs(i - j), ans)
        j += 1
print(ans)