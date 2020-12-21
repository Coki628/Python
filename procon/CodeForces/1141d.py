# -*- coding: utf-8 -*-

import sys
from collections import defaultdict, deque
from string import ascii_lowercase

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
L=input()
R=input()

C1=defaultdict(deque)
for i in range(N):
    C1[L[i]].append(i+1)
C2=defaultdict(deque)
for i in range(N):
    C2[R[i]].append(i+1)

ans=[]
for s in ascii_lowercase:
    while len(C1[s]) and len(C2[s]):
        a=C1[s].popleft()
        b=C2[s].popleft()
        ans.append((a, b))

for s in ascii_lowercase:
    while len(C1['?']) and len(C2[s]):
        a=C1['?'].popleft()
        b=C2[s].popleft()
        ans.append((a, b))

for s in ascii_lowercase:
    while len(C2['?']) and len(C1[s]):
        b=C2['?'].popleft()
        a=C1[s].popleft()
        ans.append((a, b))

while len(C1['?']) and len(C2['?']):
    a=C1['?'].popleft()
    b=C2['?'].popleft()
    ans.append((a, b))

print(len(ans))
for v in ans:
    print(*v)
