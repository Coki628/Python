"""
参考：https://qiita.com/Kiri8128/items/eca965fe86ea5f4cbb98
・素因数分解(高速版)、約数の個数
・Kiriさんの記事から。素因数分解の高速なやつ。
・でもこれはさすがに愚直を落とすためみたいな制約なので通らなかったー。。
　(なんか実際のKiriさんの提出を見てみると、これ+埋め込みみたいなのを駆使して通してた。。)
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def factorize(N):
    def gcd(a, b):
        while b: a, b = b, a % b
        return a
    def isPrimeMR(n):
        d = n - 1
        d = d // (d & -d)
        L = [2]
        for a in L:
            t = d
            y = pow(a, t, n)
            if y == 1: continue
            while y != n - 1:
                y = (y * y) % n
                if y == 1 or t == n - 1: return 0
                t <<= 1
        return 1
    def findFactorRho(n):
        m = 1 << n.bit_length() // 8
        for c in range(1, 99):
            f = lambda x: (x * x + c) % n
            y, r, q, g = 2, 1, 1, 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = f(y)
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = f(y)
                        q = q * abs(x - y) % n
                    g = gcd(q, n)
                    k += m
                r <<= 1
            if g == n:
                g = 1
                while g == 1:
                    ys = f(ys)
                    g = gcd(abs(x - ys), n)
            if g < n:
                if isPrimeMR(g): return g
                elif isPrimeMR(n // g): return n // g
                return findFactorRho(g)
    def primeFactor(n):
        i = 2
        ret = {}
        rhoFlg = 0
        while i*i <= n:
            k = 0
            while n % i == 0:
                n //= i
                k += 1
            if k: ret[i] = k
            i += 1 + i % 2
            if i == 101 and n >= 2 ** 20:
                while n > 1:
                    if isPrimeMR(n):
                        ret[n], n = 1, 1
                    else:
                        rhoFlg = 1
                        j = findFactorRho(n)
                        k = 0
                        while n % j == 0:
                            n //= j
                            k += 1
                        ret[j] = k
        if n > 1: ret[n] = 1
        if rhoFlg: ret = {x: ret[x] for x in sorted(ret)}
        return ret
    return primeFactor(N)

N = INT()

ans = 0
for i in range(1, N+1):
    res = factorize(i)
    cnt = 1
    for v in res.values():
        cnt *= v + 1
    ans += cnt * i
print(ans)
