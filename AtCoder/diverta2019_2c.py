# -*- coding: utf-8 -*-

import sys
from collections import deque

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N=INT()
A=deque(sorted(LIST()))

ans=[]
while len(A)>=3:
    if A[0]<=0 and A[-1]>=0:
        if A[1]<=0:
            y=A.popleft()
            x=A.pop()
            ans.append((x, y))
            A.append(x-y)
        elif A[-2]>=0:
            x=A.popleft()
            y=A.pop()
            ans.append((x, y))
            A.appendleft(x-y)
        else:
            if A[0]>0:
                x=A.popleft()
                y=A.pop()
                ans.append((x, y))
                A.appendleft(x-y)
            elif A[-1]<0:       
                y=A.popleft()
                x=A.pop()
                ans.append((x, y))
                A.append(x-y)
            else:
                raise Exception
    else:
        if A[0]>0:
            x=A.popleft()
            y=A.pop()
            ans.append((x, y))
            A.appendleft(x-y)
        elif A[-1]<0:       
            y=A.popleft()
            x=A.pop()
            ans.append((x, y))
            A.append(x-y)
        else:
            raise Exception
ans.append((A[-1], A[0]))

print(A[-1]-A[0])
for x, y in ans:
    print(x, y)
