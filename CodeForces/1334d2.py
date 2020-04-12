# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/laycrs/status/1248654727912378368
・自力ならず。。
・グラフ、構築、辞書順最小、貪欲
・辞書順最小の作り方がミスってた。。
・とにかく小さい数字を前に出すのが最優先なので、12131415,232425,3435,45,1 みたいになる。
　そりゃそうだよね、1は全部先に使うべき。最後のやつ以外。
・自分のWAしてた実装を流れはそのままで、作り方だけ直したら無事AC。
"""

import sys
from itertools import accumulate

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

# 今回分(v,v+1,v,v+2,...)を構築する
def gen_arr(v):
    # 最後だけ例外処理
    if N - v == 0:
        return [1]
    n = (N-v) * 2
    res = [0] * n
    x = v + 1
    for i in range(n):
        if i % 2 == 0:
            res[i] = v
        else:
            res[i] = x
            x += 1
    return res

for _ in range(INT()):
    N, l, r = MAP()
    l -= 1

    ans = []
    # cur := 現在の先頭index
    cur = 0
    # v := 今回軸にする頂点
    v = 1
    # incr := 今回の構築での増分
    incr = (N-1) * 2
    while cur + incr < l:
        cur += incr
        v += 1
        incr -= 2
    ans = gen_arr(v)
    # 左端より後ろだけ切り出す
    ans = ans[l-cur:]
    ln = r - l
    v += 1
    while len(ans) < ln:
        ans += gen_arr(v)
        v += 1
    # 右端より前だけ切り出す
    ans = ans[:ln]
    print(*ans)
