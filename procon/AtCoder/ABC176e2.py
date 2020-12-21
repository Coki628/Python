"""
参考：https://atcoder.jp/contests/abc176/editorial/66
・公式解
・鳩の巣原理
・実は元々行maxかつ列maxのとこにするのが常に最善で、その中で交差するマスに爆破対象があるかないかで
　-1するかしないか、になる。爆破対象がないマスが1つでもあれば-1しなくて済むし、なければ-1する。
・行maxのとこ*列maxのとこで愚直に2重ループ回さないといけないが、
　最大ケースで調べる対象がH*Wくらいあったとしても、そのうち爆破対象があるマスはM個しかないので、
　候補が多い場合、M回箇所くらい見たら絶対爆破対象のない場所に当たる。
　ここで探索を打ち切れば、H*Wを全部見るような状況はない。(鳩の巣原理)
・ここまで考察できると平面走査しなくても通せる。速度も早くてpypyAC0.33秒。
・全体としてどっちで通した人が多いんだろ。こっちに行ける人頭いいなー、とは思う。
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
MOD = 10 ** 19 + 7
EPS = 10 ** -10

H, W, M = MAP()
row = [0] * (H+1)
col = [0] * (W+1)
se = set()
for i in range(M):
    h, w = MAP()
    row[h] += 1
    col[w] += 1
    se.add((h, w))

mxh = max(row)
hli = []
for h, cnt in enumerate(row):
    if cnt == mxh:
        hli.append(h)
mxw = max(col)
wli = []
for w, cnt in enumerate(col):
    if cnt == mxw:
        wli.append(w)

ok = 0
for h in hli:
    for w in wli:
        if (h, w) not in se:
            ok = 1
            break
    if ok:
        break
ans = mxh + mxw - 1 + ok
print(ans)
