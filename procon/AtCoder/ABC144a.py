def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]

A, B = MAP()

if A < 10 and B < 10:
    print(A*B)
else:
    print(-1)
