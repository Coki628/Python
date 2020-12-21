# -*- coding: utf-8 -*-

str_list = list(input())

num = 0
for str in str_list:
    if str == "+":
        num += 1
    elif str == "-":
        num -= 1
print(num)