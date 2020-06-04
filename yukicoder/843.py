"""
・自力AC！
・素数列挙、候補を減らして全探索、式変形
・減らしたとはいえ、最大ケースN=50万で41538*168=約700万くらい。pythonAC1.5秒。
・ちなみに公式解説は、足し算掛け算の結果の偶奇を利用して、p,qどちらかは2しか使えないとして
　候補を大幅に絞り込んでた。偶奇に注目するのは大事だし、そっちの考察もできるといいね。。
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

def eratosthenes_sieve(n):
    """ 素数列挙(エラトステネスの篩) """

    table = [0] * (n + 1)
    prime_list = []
    for i in range(2, n + 1):
        if table[i] == 0:
            prime_list.append(i)
            for j in range(i + i, n + 1, i):
                table[j] = 1
    return prime_list

N = INT()
primes = eratosthenes_sieve(N)
S = set(primes)

# rの候補を絞り込む
R = []
for p in primes:
    # p+qが取りうるのはせいぜいN*2くらいまで(p,qともN以下なので)
    if p**2 > N*2:
        break
    R.append(p**2)

ans = 0
for p in primes:
    for r2 in R:
        q = r2 - p
        # 式変形して、qが素数リストに入ってるならOK
        if q in S:
            ans += 1
print(ans)
