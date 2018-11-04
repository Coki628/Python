# -*- coding: utf-8 -*-

N = int(input())

if N <= 59:
    print('Bad')
elif 60 <= N <= 89:
    print('Good')
elif 90 <= N <= 99:
    print('Great')
elif N == 100:
    print('Perfect')