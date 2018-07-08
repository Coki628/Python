# -*- coding: utf-8 -*-

ABC = list(map(int, input().split()))

# ABCのどれなのかは関係ないので昇順ソートしとく
ABC.sort()
# まず、少ないやつを差が1になるまで+2していく
cnt = 0
while ABC[0] <= ABC[2] - 2:
	ABC[0] += 2
	cnt += 1
while ABC[1] <= ABC[2] - 2:
	ABC[1] += 2
	cnt += 1
# あとは3つのパターンから場合分け
if ABC[0] == ABC[1] == ABC[2]:
	print(cnt)
	exit()
elif ABC[0] == ABC[2] or ABC[1] == ABC[2]:
	cnt += 2
	print(cnt)
	exit()
elif ABC[0] == ABC[1]:
	cnt += 1
	print(cnt)
	exit()