"""
参考：https://ja.wikipedia.org/wiki/%E5%AE%9A%E8%A6%8F%E3%81%A8%E3%82%B3%E3%83%B3%E3%83%91%E3%82%B9%E3%81%AB%E3%82%88%E3%82%8B%E4%BD%9C%E5%9B%B3
　　　https://ja.wikipedia.org/wiki/%E3%83%95%E3%82%A7%E3%83%AB%E3%83%9E%E3%83%BC%E6%95%B0#%E3%83%95%E3%82%A7%E3%83%AB%E3%83%9E%E3%83%BC%E7%B4%A0%E6%95%B0
・自力AC
・数学、ビット全探索に帰着
・なんていうかevimaさんぽい教育的な問題というか。。
・定規とコンパスで作図？フェルマー素数？ってなるけど、ようは5通りビット全探索。
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

N = INT()

# 10^9までの2冪
A = []
k = 0
while 2**k <= 10**9:
    A.append(2**k)
    k += 1
# フェルマー素数
B = [3, 5, 17, 257, 65537]

# 全組み合わせ調べればいい
ans = 0
for a in A:
    for bit in range(1<<5):
        cur = a
        for i in range(5):
            if bit >> i & 1:
                cur *= B[i]
        if 3 <= cur <= N:
            ans += 1
print(ans)
