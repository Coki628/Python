"""
参考：https://www.hamayanhamayan.com/entry/2020/08/01/122317
　　　https://yukicoder.me/problems/no/1140/editorial
・あまりにインチキACなので解説ちゃんと読んだよ。
・素数列挙、式変形、冪乗の階乗？
・まあ素数列挙は空気。本題はA^(P!) mod Pみたいな謎のでかいやつを解くところだ。
・Pが素数なので、フェルマーみたいな何かをうまいこと使うんかなぁとかググったけどいい情報は出ず。
・実際フェルマーだったんだけど、いつものMOD逆元のMOD-2の形じゃなくて、MOD-1=1ってやつを使う。
　(多分こっちのが一般に言うフェルマーの小定理なんだったと思う。)
・これを使って式変形するんだけど、もう1つ自分が知らなかったことが。
　a^(b*c)みたいな時、(a^b)^cって分解できる。へー、知らんかった。
　a^(b+c)だとa^b*a^cってできるけど、指数が掛け算の時ってバラせないなぁって昔考えたことはあった。バラせた。
・ここから、a^(p-1)^(p*(p-2)!)みたいにしてa^(p-1)はmod pで1なので、1は何乗しても1、ってなる。
　でもaがpの倍数だったりするとフェルマーのルールが使えないから、
　その時はa何回掛けてもpで割り切れるから0、ってなる。数学むずい。
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

def eratosthenes_sieve(n):
    """ 素数列挙(エラトステネスの篩) """

    table = [0] * (n+1)
    prime_list = []
    for i in range(2, n+1):
        if table[i] == 0:
            prime_list.append(i)
            for j in range(i+i, n+1, i):
                table[j] = 1
    return prime_list

primes = set(eratosthenes_sieve(5000000))

for _ in range(INT()):
    a, p = MAP()

    if p in primes:
        if a % p == 0:
            print(0)
        else:
            print(1)
    else:
        print(-1)
