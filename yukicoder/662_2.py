"""
参考：https://www.hamayanhamayan.com/entry/2018/03/10/213134
・想定解でもっと効率よく解かれていたので。
・とはいえ、この*5すれば帳尻合うのはなんか非自明だよなー。。
　言われれば確かにそうかも、ではあるんだけどさ。
"""

import sys
from collections import Counter

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

M = 5
coin = [0] * M
stoi = {}
for i in range(M):
    s, c = input().split()
    c = int(c)
    coin[i] = c
    stoi[s] = i

C = [Counter() for i in range(3)]
for i in range(3):
    N = INT()
    S = [input() for i in range(N)]
    for j in range(N):
        # 列毎に、絵柄の出現回数を集計する
        C[i][stoi[S[j]]] += 1

grid = list2d(3, 3, 0)
ans = [0] * M
total = 0
sm = 0
# スロットの3つの列を3重ループで全通り回す
for k1, v1 in C[0].items():
    for k2, v2 in C[1].items():
        for k3, v3 in C[2].items():
            cnt = v1 * v2 * v3
            if k1 == k2 == k3:
                # 当たりは横3行と斜め2つあるので*5する
                sm += coin[k1] * cnt * 5
                ans[k1] += cnt * 5
            # こっちは並びではなく画面の出方のパターンなので*5しない
            total += cnt
exp = sm / total
print(exp)
for a in ans:
    print(a)
