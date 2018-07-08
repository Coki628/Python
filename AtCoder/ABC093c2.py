# -*- coding: utf-8 -*-

ABC = list(map(int, input().split()))

maxABC = max(ABC)
sumABC = sum(ABC)

# 常に合計は+2されていくので偶奇が変化しないことを活用する
if maxABC % 2 != sumABC % 2:
	# 偶奇が合わない時は最大値でぴったりいかないので+1
	maxABC += 1

ans = 0
ans += maxABC - ABC[0]
ans += maxABC - ABC[1]
ans += maxABC - ABC[2]

print(ans // 2)