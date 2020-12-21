def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]

A = LIST()
B = LIST()

print(max(sum(A), sum(B)))
