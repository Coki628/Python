# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/75082?#comment-591995
・自力ならず。。
・数え上げ
・色々ソースコード見ても全然方針見えなくて、英語で説明してくれてる人のコメント読んでやっと理解した。
・ようは数え上げ。ちゃんとそれぞれの通り数がどうなるかを掛け合わせていくといい。
　端以外と端は場合分けしてやる。
・ソース内コメント及び自分ノートも参照。
・みんなこういうのさくっと解いててすげぇなぁ。数学地盤の差がまじででかいんだろうな。
・数え上げはあっとこでもこどふぉでもよく出るし、しっかり解けるようにしていきたい。
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
MOD = 998244353

N = INT()

# ans[i] := i個続きのブロックの通り数
ans = [0] * (N+1)
ans[N] = 10
for i in range(1, N):
    cnt = 0
    # 端以外
    if N-i-2 >= 0:
        # 連続させる数の選び方 * 隣り合う数の選び方 * 関係ない外側の選び方 * 位置の選び方
        cnt += 10 * 9 * 9 * pow(10, N-i-2, MOD) * (N-i-1)
    # 端
    cnt += 10 * 9 * pow(10, N-i-1, MOD) * 2
    ans[i] = cnt % MOD
print(*ans[1:])
