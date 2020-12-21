# -*- coding: utf-8 -*-

N = int(input())

# 等差数列の公式でO(1)
ans = N / 2 * (1 + N)
print(int(ans / N * 10000))
