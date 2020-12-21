# -*- coding: utf-8 -*-

N = input()

for i in range(1,len(N)):
    if N[i] != N[0]:
        print('DIFFERENT')
        exit()
print('SAME')