 # -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

same = False
same_cnt = 0
cnt = 0
for i in range(0, N-1):

	if same == False:
		same_cnt = 0

	if aN[i] == aN[i+1]:
		same = True
		same_cnt += 1
		if i == N-2:
			cnt += (same_cnt + 1) // 2
			break
	else:
		same = False
		if same_cnt != 0:
		 	cnt += (same_cnt + 1) // 2

print(cnt)