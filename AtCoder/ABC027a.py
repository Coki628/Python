# -*- coding: utf-8 -*-

l1,l2,l3 = map(int, input().split())

if l1 == l2 == l3:
    print(l1)
else:
    if l1 == l2:
        print(l3)
    elif l2 == l3:
        print(l1)
    elif l1 == l3:
        print(l2)