# -*- coding: utf-8 -*-

N,A,B = map(int, input().split())

pos = 0
for i in range(N):
    s,d = map(str, input().split())
    d = int(d)
    d = min(d, B)
    d = max(d, A)
    if s == 'East':
        pos += d
    elif s == 'West':
        pos -= d

if pos == 0:
    print(pos)
elif pos > 0:
    print('East ' + str(pos))
elif pos < 0:
    print('West ' + str(abs(pos)))