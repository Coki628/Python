# -*- coding: utf-8 -*-

S = list(input())

sorted_list = sorted(list(set(S)))
abc_list = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

for str1 in abc_list:
    if str1 not in sorted_list:
        print(str1)
        exit()
print("None")