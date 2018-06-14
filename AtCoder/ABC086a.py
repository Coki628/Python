# -*- coding: utf-8 -*-

num_list = [int(i) for i in input().split()]

if num_list[0] % 2 == 0 or num_list[1] % 2 == 0:
    print("Even")
else:
    print("Odd")