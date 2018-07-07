# -*- coding: utf-8 -*-
 
n, m, d = map(int, input().split())

# 条件に当てはまる(美しさが+1される)確率 * 2項の組の数
if d == 0:
    print(1 / n * (m - 1))

else:
    print((2 * (n - d) / n ** 2) * (m - 1))