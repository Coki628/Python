"""
・蟻本演習4-2-6
・Grundy数、木上のゲーム
・なんと自力AC！通ってびっくり。。
・木じゃないけど、別の問題で再帰で末端まで行って、戻り時に各Grundy数をマージしてくみたいなやつがあって、
　同じ要領で部分木のGrundy数マージしてって根まで戻せばいけるんじゃないかなって思って、
　サンプル手元で試したら合ってたから、ちゃちゃっと書いてダメ元で投げたらAC！
・ちなみに10万の木だけど、pypyはMLEするしTLEするしでひどかった。pythonでAC0.40秒。
　昔のpypyもうちょっと頑張れた気がするけど。。やっぱ再帰は基本C++だな。
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
sys.setrecursionlimit(10**7)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N = INT()
nodes = [[] for i in range(N)]
for _ in range(N-1):
    u, v = MAP()
    u -= 1; v -= 1
    nodes[u].append(v)
    nodes[v].append(u)

def dfs(u, prev):
    res = 0
    for v in nodes[u]:
        if v == prev:
            continue
        res ^= dfs(v, u) + 1
    return res
res = dfs(0, -1)
if res:
    print('Alice')
else:
    print('Bob')
