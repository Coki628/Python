def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]

N = INT()

S = set()
for i in range(1, 10):
    for j in range(1, 10):
        S.add(i*j)
if N in S:
    print('Yes')
else:
    print('No')
