"""
・自力ならず。。
・ffoxofoxxみたいなケースが考慮できてなかった。。
・公式はスタックで賢く、みたいにしてたけど、思いつける自信もなかったので、
　C++でmap愚直シミュで通しておいた。。
"""

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

N = INT()
S = input()

cnt = 0
i = 0
while i < N-2:
    if S[i:i+3] == 'fox':
        cnt += 3
        l = i
        r = i + 3
        while 1:
            if l-2 >= 0 and r < N and \
                    S[l-2] == 'f' and S[l-1] == 'o' and S[r] == 'x':
                l = l - 2
                r = r + 1
                cnt += 3
            elif l-1 >= 0 and r+1 < N and \
                    S[l-1] == 'f' and S[r] == 'o' and S[r+1] == 'x':
                l = l - 1
                r = r + 2
                cnt += 3
            else:
                break
        i = r
    else:
        i += 1
ans = N - cnt
print(ans)
