 # -*- coding: utf-8 -*-

T = int(input())

for i in range(T):
	A, B, C, D = map(int, input().split())
	zaiko = A
	while True:
		zaiko -= B
		if zaiko < 0:
			print("No")
			break
		if zaiko <= C:
			zaiko += D
			if zaiko == A:
				print("Yes")
				break