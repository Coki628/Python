# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

d = {}
for i in range(N):
    d[i+1] = aN[i]

# dictをvalueで降順ソート
items = sorted(d.items(), key=lambda x: x[1], reverse=True)

for item in items:
    print(item[0])