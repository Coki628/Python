"""
・自力AC！
・ゲーム系、インタラクティブ
・とりあえず考察詰まったのでゲームDPを作って状態を列挙。負け状態に何か特徴がないか見る。
・すると初手でできる動きとして真ん中で区切るような操作をしており、
　その先も左右に同数残ったような状態は負け続けるということが分かってくる。
・ここから、最初に半分に割ったあと、相手と反対側で同じことを続ければ、
　いずれ全部埋まった状態を相手に渡すことができそう、となる。これでAC。
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

# N = INT()
# S = '0' * N

# memo = {}
# def rec(S):
#     if S in memo:
#         return memo[S]
#     if not S.count('0'):
#         memo[S] = 0
#         return 0
#     S = list(S)
#     res = 1
#     for i in range(N):
#         if S[i] == '0':
#             S[i] = '1'
#             res &= rec(''.join(S))
#             S[i] = '0'
#     for i in range(N-1):
#         if S[i] == S[i+1] == '0':
#             S[i] = S[i+1] = '1'
#             res &= rec(''.join(S))
#             S[i] = S[i+1] = '0'
#     S = ''.join(S)
#     memo[S] = 1-res
#     return 1-res

# res = rec(S)
# win = []
# for k, v in memo.items():
#     # 負け状態を列挙して調べる
#     if not v:
#         win.append(k)
# win.reverse()
# for s in win:
#     if s.count('1') <= 4:
#         print(s.count('1'), s)

N = INT()
S = ['0'] * N

# 真ん中で区切る
if N % 2 == 0:
    print('{0} {1}'.format(2, N//2), flush=1)
    S[N//2-1] = S[N//2] = '1' 
else:
    print('{0} {1}'.format(1, ceil(N, 2)), flush=1)
    S[ceil(N, 2)-1] = '1'

cur = ['0'] * (ceil(N, 2)-1)
dist = N//2 + 1
while 1:
    t = INT()
    if t == 0:
        break

    k, x = MAP()
    x -= 1
    for i in range(x, x+k):
        S[x] = '1'
    if N % 2 == 0:
        L, R = S[:N//2-1], S[N//2+1:]
    else:
        L, R = S[:ceil(N, 2)-1], S[ceil(N, 2):]
    # 左右反対側で同じことをする
    if L != cur:
        x += dist
        cur = L
    else:
        x -= dist
        cur = R
    for i in range(x, x+k):
        S[x] = '1'
    x += 1
    print('{0} {1}'.format(k, x), flush=1)
