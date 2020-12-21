# -*- coding: utf-8 -*-

N = int(input())
AN = list(map(int, input().split()))

AN.sort()

print(AN[-1] - AN[0])