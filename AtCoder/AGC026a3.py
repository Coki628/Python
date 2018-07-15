 # -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

# 実はこんなシンプルで良かったなんて…
cnt = 0
for i in range(1, N):
	if aN[i] == aN[i-1]:
		cnt += 1
		# 左右と被らない数、を実際に探さないで-1ってことにしとく
		aN[i] = -1
print(cnt)