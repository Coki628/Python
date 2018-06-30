# -*- coding: utf-8 -*-

N = int(input())
tests = list(map(int, input().split()))
M = int(input())

# 予め基準になる合計値を出しておく
sum1 = sum(tests)

for i in range(M):
    p, x = map(int, input().split())
    # 先に出しておいた基準からの差分で算出
    print(sum1 + (x - tests[p-1]))