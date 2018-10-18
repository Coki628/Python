# -*- coding: utf-8 -*-

n = input()

n = n.replace('1','a')
n = n.replace('9','b')
n = n.replace('a','9')
n = n.replace('b','1')

print(n)