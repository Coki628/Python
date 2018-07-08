# -*- coding: utf-8 -*-

N = int(input())

# 出発点と終着点もリストに追加しておく
AN = [0]
AN += list(map(int, input().split()))
AN += [0]

# 全地点行った場合の各距離と合計距離を出しておく
All = []
for i in range(N+1):
	All.append(abs(AN[i] - AN[i+1]))
dist = sum(All)

# 行かなくなった箇所に応じた距離の増減をして出力
for i in range(1, N+1):
	tmp = dist
	tmp -= All[i-1] + All[i]
	tmp += abs(AN[i-1] - AN[i+1])
	print(tmp)