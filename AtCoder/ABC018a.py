# -*- coding: utf-8 -*-

l = []
l.append([int(input()), 1])
l.append([int(input()), 2])
l.append([int(input()), 3])
l.sort(reverse=True)
for i in range(3):
    l[i].append(i+1)
l.sort(key=lambda x: x[1])
for i in range(3):
    print(l[i][2])
