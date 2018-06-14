# -*- coding: utf-8 -*-

# 入力を各変数に格納
n, m = map(int, input().split())

ans = 0

# 行列ともに1
if n == 1 and m == 1:
    ans = 1
# 行が1
elif n == 1:
    ans = m - 2
# 列が1
elif m == 1:
    ans = n - 2
# 行列とも1ではない
else:
    ans = (n - 2) * (m - 2)

print(ans)