"""
・場合分け地獄。まじここで消耗したんだけど、速い人どうやってんだろ。。
"""

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N, K = MAP()

    if N < K:
        NO()
        continue
    d, m = divmod(N, K)
    if K % 2 == 0 and m % 2 == 1:
        NO()
        continue
    ans = [d] * K
    if m % 2 == 0:
        ans[0] += m
    else:
        for i in range(K-1):
            if i % 2 == 0:
                ans[i] += 1
            else:
                ans[i] -= 1
        ans[-1] += 1
        m -= 1
        if d == 1:
            i = 1
            for i in range(1, N, 2):
                if m <= 0:
                    break
                ans[i] += 2
                m -= 2
            ans[-1] += m
            if min(ans) == 0:
                NO()
                continue
        else:
            ans[-1] += m
    YES()
    print(*ans)
