# -*- coding: utf-8 -*-

n = int(input())

ans = float('inf')

# 正方形
i = 1
while i * i <= n:
    # 全体 - 正方形
    ans = n - i * i
    i += 1

# 長方形
j = 0
# 短辺を1ずつ短くするループ
while i > 0:
    # 長辺を面積がnを超えるまで長くするループ
    while (i) * (i+j) <= n:
        # 全体 - 長方形 + 長辺と短辺の差分
        tmp = n - (i)*(i+j) + j
        ans = min(tmp, ans)
        j += 1
    i -= 1

print(ans)