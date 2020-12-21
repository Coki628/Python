import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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
    H, W = MAP()
    grid = [[]] * H
    for i in range(H):
        grid[i] = list(input())

    ans = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '1':
                if i == 0 and j == 0:
                    ans.append((i+1, j+1, i+1, j+2, i+2, j+1))
                    ans.append((i+1, j+1, i+1, j+2, i+2, j+2))
                    ans.append((i+1, j+1, i+2, j+1, i+2, j+2))
                elif i == 0:
                    ans.append((i+1, j+1, i+1, j+0, i+2, j+1))
                    ans.append((i+1, j+1, i+1, j+0, i+2, j+0))
                    ans.append((i+1, j+1, i+2, j+1, i+2, j+0))
                elif j == 0:
                    ans.append((i+1, j+1, i+0, j+1, i+1, j+2))
                    ans.append((i+0, j+1, i+1, j+1, i+0, j+2))
                    ans.append((i+1, j+1, i+1, j+2, i+0, j+2))
                else:
                    ans.append((i+1, j+1, i+0, j+1, i+1, j+0))
                    ans.append((i+0, j+0, i+0, j+1, i+1, j+1))
                    ans.append((i+0, j+0, i+1, j+0, i+1, j+1))
    print(len(ans))
    for a in ans:
        print(*a)
