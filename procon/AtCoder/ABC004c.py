# -*- coding: utf-8 -*-

N = int(input())

# 30で循環してるぽいのでmod 30
N %= 30

nums = [0] * 6
for i in range(6):
    nums[i] = i+1

for i in range(N):
    # swap
    nums[i%5], nums[i%5+1] = nums[i%5+1], nums[i%5]

for i in range(6):
    nums[i] = str(nums[i])
print(''.join(nums))
