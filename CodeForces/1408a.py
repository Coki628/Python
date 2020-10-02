import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N = INT()
    A = LIST()
    B = LIST()
    C = LIST()

    ans = [A[0]]
    for i in range(1, N-1):
        if ans[-1] != A[i]:
            ans.append(A[i])
        else:
            ans.append(B[i])
    if ans[-1] != A[-1] and A[-1] != ans[0]:
        ans.append(A[-1])
    elif ans[-1] != B[-1] and B[-1] != ans[0]:
        ans.append(B[-1])
    else:
        ans.append(C[-1])

    print(*ans)
