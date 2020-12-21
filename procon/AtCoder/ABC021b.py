# -*- coding: utf-8 -*-

N = int(input())
a,b = map(int, input().split())
K = int(input())
pK = list(map(int, input().split()))

pK.append(a)
pK.append(b)

if len(pK) == len(set(pK)):
    print('YES')
else:
    print('NO')
