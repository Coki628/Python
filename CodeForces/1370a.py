"""
・なんとか通せた。
・素数列挙応用、前計算
・ある数xとペアを組ませてGCDを最大化できる値は、「x / xを割り切れる最小の値」で求まる。
・「xを割り切れる最小の値」は素数テーブル埋めに行く時の素数なので、素数列挙でそれを保持して計算。
・あとはクエリにO(1)で対応できるように下から累積max取っておけばOK。
→やばいこれ。他の人の見たら「N割る2」してるだけなんだが。。考えたら確かに。。。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

def eratosthenes_sieve(n):
    table = [1] * (n+1)
    for i in range(2, n+1):
        if table[i] == 1:
            for j in range(i+i, n+1, i):
                if table[j] == 1:
                    table[j] = i
    for i in range(2, n+1):
        if table[i] != 1:
            table[i] = i // table[i]
    for i in range(2, n+1):
        table[i] = max(table[i], table[i-1])
    return table

table = eratosthenes_sieve(10**6)

for _ in range(INT()):
    N = INT()

    ans = table[N]
    print(ans)
