# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/69954
・自力ならず。。
・1ブロックずつ前に進んで、該当のブロックに着いたら今度は値を1つずつ進める。
・言えばそれだけなんだけど、進むに当たって持つ変数が色々あって結構ややこしい。
　解説読んでもよく分かんなくて、例のコード実際に動かしてやっと何やってるか理解した。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    K = INT()
    done = 0
    # cnt := そこまでの合計文字数
    cnt = 0
    # blkln := 今回追加するブロックの長さ
    blkln = 0
    # digln := 今回新たに増える数字の桁数
    digln = 0
    i = 1
    # nxpow := 次に桁が増えるindex
    nxpow = 1
    # まずはブロック毎まとめて追加していく
    while not done:
        if i == nxpow:
            nxpow *= 10
            digln += 1
        blkln += digln
        if cnt + blkln < K:
            cnt += blkln
        else:
            digln = 0
            j = 1
            nxpow = 1
            # 目的のブロックに着いたら、値を1つずつ追加していく
            while 1:
                if j == nxpow:
                    nxpow *= 10
                    digln += 1
                if cnt + digln < K:
                    cnt += digln
                else:
                    num = str(j)
                    ans = num[K-cnt-1]
                    print(ans)
                    done = 1
                    break
                j += 1
        i += 1
