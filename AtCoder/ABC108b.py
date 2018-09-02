# -*- coding: utf-8 -*-

x1,y1,x2,y2 = map(int, input().split())

X = x2 - x1
Y = y2 - y1

if X > 0 and Y > 0:
    x3 = x2 - abs(Y)
    y3 = y2 + abs(X)
    x4 = x3 - abs(X)
    y4 = y3 - abs(Y)

elif X > 0 and Y < 0:
    x3 = x2 + abs(Y)
    y3 = y2 + abs(X)
    x4 = x3 - abs(X)
    y4 = y3 + abs(Y)
    
elif X < 0 and Y > 0:
    x3 = x2 - abs(Y)
    y3 = y2 - abs(X)
    x4 = x3 + abs(X)
    y4 = y3 - abs(Y)

elif X < 0 and Y < 0:
    x3 = x2 + abs(Y)
    y3 = y2 - abs(X)
    x4 = x3 + abs(X)
    y4 = y3 + abs(Y)

elif X > 0 and Y == 0:
    x3 = x2
    y3 = y2 + abs(X)
    x4 = x3 - abs(X)
    y4 = y3

elif X == 0 and Y > 0:
    x3 = x2 - abs(Y)
    y3 = y2
    x4 = x3
    y4 = y3 - abs(Y)

elif X < 0 and Y == 0:
    x3 = x2
    y3 = y2 - abs(X)
    x4 = x3 + abs(X)
    y4 = y3

elif X == 0 and Y < 0:
    x3 = x2 + abs(Y)
    y3 = y2
    x4 = x3
    y4 = y3 + abs(Y)

print(str(x3) + ' ' + str(y3) + ' ' + str(x4) + ' ' + str(y4))