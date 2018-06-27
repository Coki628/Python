# -*- coding: utf-8 -*-

x, y = map(int, input().split())

# グループを識別する値を入れておく
a = [0,1,3,1,2,1,2,1,1,2,1,2,1]

if a[x] == a[y]:
    print("Yes")
else:
    print("No")