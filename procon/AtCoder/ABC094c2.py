# -*- coding: utf-8 -*-

"""
成長の確認のため、昔やったやつを自力ACしてみる
"""

N = int(input())
aN = list(map(int, input().split()))

aN2 = sorted(aN)

if aN2[N//2-1] == aN2[N//2]:
	for i in range(N):
		print(aN2[N//2])
else:
	for i in range(N):
		if aN[i] < aN2[N//2]:
			print(aN2[N//2])
		else:
			print(aN2[N//2-1])
