# -*- coding: utf-8 -*-

N = int(input())

num_list = [1]
for i in [6, 9]:
    j = 1
    while True:
        if i ** j <= 100000:
            num_list.append(i ** j)
        else:
            break
        j += 1

num_list.sort(reverse=True)

sum1 = 0
cnt = 0
for num in num_list:
    if sum1 + num <= N:
        cnt += (N - sum1) // num
        sum1 += num * ((N - sum1) // num)

print(cnt)