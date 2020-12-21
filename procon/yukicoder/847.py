"""
・自力ならず。。
・素因数分解、枝刈り全探索、DFS、バックトラック
・素因数分解して冪乗の分掛けるまではやったんだけど、その先に進めず。。
　大きくなりすぎた所で打ち切り、は考えたけど、うまいやり方が出ず。
・正解はDFSで枝刈り。BFSでもできなくはないかもだけど、
　バックトラックで状態戻しがしやすいので、再帰DFSがよさそう。
・公式解説より、最大ケースは10万くらいとのことだが、
　pythonはギリギリAC1.9秒、pypyは再帰弱くてだめかと思ったらAC0.4秒。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def factorize(num: int) -> dict:
    """ 素因数分解 """
    from math import sqrt
    from collections import Counter

    d = Counter()
    for i in range(2, int(sqrt(num))+1):
        while num % i == 0:
            num //= i
            d[i] += 1
        if num == 1:
            break
    if num != 1:
        d[num] += 1
    return d

def dfs(u):
    for key, val in primes.items():
        for k in range(1, val+1):
            v = u * key**k
            if v > M:
                break
            if v in visited:
                continue
            visited.add(v)
            primes[key] -= 1
            dfs(v)
            primes[key] += 1

N, K, M = MAP()

primes = factorize(N)
for k in primes.keys():
    primes[k] *= K

visited = set()
visited.add(1)
dfs(1)
ans = len(visited)
print(ans)
