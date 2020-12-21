# -*- coding: utf-8 -*-

N = input()

for i in range(10):

    if str(i) + str(i) + str(i) in N:
        print("Yes")
        exit()
print("No")