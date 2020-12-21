import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    A = LIST()

    idx1 = idx2 = idx3 = -1
    for i in range(N-1):
        if A[i] < A[i+1]:
            idx1 = i
        if idx1 != -1 and A[i] > A[i+1]:
            idx2 = i
            idx3 = i + 1
            break
    if idx3 != -1:
        YES()
        print(idx1+1, idx2+1, idx3+1)
    else:
        NO()
