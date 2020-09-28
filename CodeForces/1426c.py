"""
・とりあえず題意の操作を5回くらい進めると、操作1,2を半々くらいでやったところで最適だなぁとなる。
・その最適っぽい値がどう変化していくか見ると、1,2,4,6,9,12,16,20みたいになるので、
　2回に1回、加算分を増やしながら、前の数に加算分を足していく感じだなぁとなる。
・で、実装したけど合わない。なんでかなーと思ったけど、結局例外処理してたN=2のとこが間違ってた。。
　終了後ちゃんと見たら、別に例外処理しなくても規則的じゃん最初から、ってなって整理したのがこれ。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N = INT()

    cnt = 0
    cur = add = 1
    while cur < N:
        cur += add
        cnt += 1
        if cnt % 2 == 1:
            add += 1
    print(cnt)
