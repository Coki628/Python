"""
・自力AC。。
・wolframalpha
・きれいな式になりそうな雰囲気だったので、N=8くらいまで増やして、それぞれのiが何回使われるかを
　実験したら、8 14 18 20 20 18 14 8みたいになったので、これをwolframalphaそのまま貼ると、
　a_n = 9 n - n^2 って出たのでこれをそのまま当てはめた。
・ツールをうまく使えたって意味ではいいんだけど、素直に喜んでいいのか分からんよねこういうの。
"""

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

N = INT()
A = LIST()

# cnt = [0] * N
# for i in range(N):
#     for j in range(i, N):
#         for k in range(i, j+1):
#             cnt[k] += 1
# print(*cnt)

ans = 0
for i in range(N):
    ans += A[i] * ((N+1)*(i+1) - (i+1)**2)
print(ans)
