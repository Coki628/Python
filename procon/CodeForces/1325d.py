# -*- coding: utf-8 -*-

"""
・なんとか自力AC
・XOR系、てかXORSUM系
・あっとこでよく出てきた論点だし、なんとか解けないかと頑張った。
・最初から方針は案外合ってたのに、時間かかった。。
・結局長く詰まっていた原因は、10^18の制約に対して、最大2^18でやろうとしてたことだったんだよね。。
　こういう凡ミスなくさないとなー。。
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

u, v = MAP()

# XORのが大きいのは即NG
if v < u:
    print(-1)
    exit()

# まずXORで使う分のビットを立てる
cnt = [0] * 61
for k in range(61):
    if u & 1<<k:
        cnt[k] += 1

# 残りについて、ビットが立たないように2つずつ追加する
need = v - u
for k in range(60, -1, -1):
    if (1<<k) * 2 <= need:
        cnt[k] += 2
        need -= (1<<k) * 2

# 全部使いきれず残ってしまったらNG
if need != 0:
    print(-1)
    exit()

# 集めたビットを数列に割り当てていく
ans = [0] * max(cnt)
for k, v in enumerate(cnt):
    for i in range(v):
        ans[i] |= 1<<k
print(len(ans))
if ans:
    print(*ans)
