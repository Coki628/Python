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
    N, M = MAP()
    rows = [[]] * N
    for i in range(N):
        rows[i] = LIST()
    cols = [[]] * M
    for i in range(M):
        cols[i] = LIST()

    keycol = -1
    for i in range(M):
        for j in range(N):
            if cols[i][j] == rows[0][0]:
                keycol = i
                break
        if keycol != -1:
            break

    D = {}
    for j in range(N):
        D[cols[keycol][j]] = j
    
    ans = [[]] * N
    for i in range(N):
        ans[D[rows[i][0]]] = rows[i]

    for row in ans:
        print(*row)
