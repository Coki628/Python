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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def RLE(data):
    """ ランレングス圧縮 """
    from itertools import groupby

    return [(x, len(list(grp))) for x, grp in groupby(data)]

N = INT()
S = input()

rle = RLE(S)
ans = [0] * 2
if rle[0][0] == '.':
    ans[0] = rle[0][1]
if rle[-1][0] == '.':
    ans[1] = rle[-1][1]

mx = ans[0] + ans[1]
for s, cnt in rle[1:-1]:
    if s == '.':
        mx = max(mx, cnt)
ans[0] += mx - (ans[0] + ans[1])
print(*ans)
