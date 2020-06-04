"""
・自力AC
・工夫して集計、期待値
・スロット各列の縦の長さは最大5000もあるので3列全通りは試せないけど、
　絵柄が5種類しかないので、画面に映る縦3つの並びはそんなにパターンが多くない
　(5種から3種選ぶ*3つの並べ方で5C3*3! = 60(多分))ので、これなら3列分の3重ループが回せる。
・これで画面に映る9個の絵柄のパターンとそれが出る通り数を全て網羅できるので、
　あとは当たりになる5種類の並びがいくつあるか数えればOK。期待値はおまけ。
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
        k = (j+1) % N
        l = (j+2) % N
        # 列毎に、絵柄の縦の並びを全通り集計する
        C[i][(stoi[S[j]], stoi[S[k]], stoi[S[l]])] += 1

grid = list2d(3, 3, 0)
ans = [0] * M
total = 0
sm = 0
# スロットの3つの列を3重ループで全通り回す
for k1, v1 in C[0].items():
    for i, s in enumerate(k1):
        grid[i][0] = s
    for k2, v2 in C[1].items():
        for i, s in enumerate(k2):
            grid[i][1] = s
        for k3, v3 in C[2].items():
            cnt = v1 * v2 * v3
            for i, s in enumerate(k3):
                grid[i][2] = s
            # 横に揃う3通り
            for i in range(3):
                if grid[i][0] == grid[i][1] == grid[i][2]:
                    ans[grid[i][0]] += cnt
                    sm += coin[grid[i][0]] * cnt
            # 斜めに揃う2通り
            if grid[0][0] == grid[1][1] == grid[2][2]:
                ans[grid[0][0]] += cnt
                sm += coin[grid[0][0]] * cnt
            if grid[0][2] == grid[1][1] == grid[2][0]:
                ans[grid[0][2]] += cnt
                sm += coin[grid[0][2]] * cnt
            total += cnt
exp = sm / total
print(exp)
for a in ans:
    print(a)
