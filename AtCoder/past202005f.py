import sys
from string import ascii_lowercase

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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 19
MOD = 10 ** 9 + 7

N = INT()
S = [input() for i in range(N)]

ans = ''
for i in range(N//2):
    j = N - i - 1
    for c in ascii_lowercase:
        if S[i].count(c) and S[j].count(c):
            ans += c
            break
    else:
        print(-1)
        exit()
if N % 2 == 0:
    ans = ans + ans[::-1]
else:
    ans = ans + S[N//2][0] + ans[::-1]
print(ans)
