 # -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

skip = False
cnt = 0
for i in range(1, N-1):
	if skip:
		skip = False
		continue
	if aN[i] == aN[i-1] or aN[i] == aN[i+1]:
		cnt += 1
		skip = True
		if aN[i] == aN[i-1]:
			for j in range(1, 10001):
				if j != aN[i-1] and j != aN[i+1]:
					aN[i] = j
					break
		if aN[i] == aN[i+1]:
			for j in range(1, 10001):
				if j != aN[i-1] and j != aN[i+1]:
					aN[i] = j
					break
print(cnt)