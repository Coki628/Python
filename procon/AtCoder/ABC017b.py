# -*- coding: utf-8 -*-

X = input()
tmp = ''
for i in range(len(X)):
    if X[i] == 'c':
        if tmp != '':
            print('NO')
            exit()
        tmp = 'c'
    elif X[i] == 'h':
        if tmp != 'c':
            print('NO')
            exit()
        tmp = ''
    elif X[i] != 'o' and X[i] != 'k' and X[i] != 'u':
        print('NO')
        exit()
print('YES')
