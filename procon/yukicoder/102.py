"""
・蟻本演習4-2-6
・Grundy数
・1つの山で最後を引くと相手から得点を奪うみたいな、ややこしいルールがあるように見せてる。
　けど実は最後の山を取った人が勝ち確定。
・それが分かれば、4つの山についてGrundy数を求められればいい。
・ノーヒントで挑んでたら、気合いの13^4*52状態を持ってゲームDPみたいのやろうとして死にそう。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
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

N = 4
A = LIST()

# 使う最大値までのGrundy数を前計算
mx = max(A)
grundy = [0] * (mx+1)
grundy[0] = 0
for i in range(1, mx+1):
    se = set()
    # 1~3枚取る遷移
    for j in range(1, 4):
        if i-j >= 0:
            se.add(grundy[i-j])
    g = 0
    while g in se:
        g += 1
    grundy[i] = g

nim = 0
for a in A:
    nim ^= grundy[a]
if nim:
    print('Taro')
else:
    print('Jiro')
