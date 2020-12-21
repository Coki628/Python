# -*- coding: utf-8 -*-

N = int(input())

i = 0
while True:
    bins = list(bin(i)[2:])
    ans = 0
    for j in range(len(bins)-1, -1, -1):
        if bins[j] == "1":
            ans += (-2) ** (len(bins)-1-j)
    if ans == N:
        print("".join(bins))
        exit()
    i += 1