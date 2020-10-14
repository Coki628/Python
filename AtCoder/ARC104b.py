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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N, S = input().split()
S = list(S)
N = int(N)

D = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
for i in range(N):
    S[i] = D[S[i]]

def check(C):
    return C[0] == C[1] and C[2] == C[3]

ans = 0
for i in range(N):
    C = [0] * 4
    C[S[i]] += 1
    for j in range(i+1, N):
        ans += check(C)
        C[S[j]] += 1
    ans += check(C)
print(ans)
