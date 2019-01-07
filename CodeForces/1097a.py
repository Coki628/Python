# -*- coding: utf-8 -*-

t = input()
me = list(input().split())

for card in me:
    if card[0] == t[0] or card[1] == t[1]:
        print('YES')
        exit()
print('NO')
