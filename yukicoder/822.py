"""
・自力AC
・論理積
・例外処理して残りはあり得る範囲を全探索。
・無限に増える場合の条件ミスってて2ペナ。ちょっと時間食った。
　最終的に仮説N < Kを確かめるために、愚直に数える方で適当にでかい1万回と10万回比べた時に、
　N < Kのケースだと差分が出たから、これはずっと増え続けそうと疑う。結果、これでAC。
・こういうの、実際コンテストで出たらみんな早そうだな。。
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

N, K = MAP()

if N < K:
    print('INF')
    exit()

ans = 0
for a in range(N, N+K+1):
    for b in range(a, a+K+1):
        if a & b == N:
            ans += 1
print(ans)
