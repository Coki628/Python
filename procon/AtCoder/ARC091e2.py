# -*- coding: utf-8 -*-

"""
参考：https://www.dropbox.com/sh/arnpe0ef5wds8cv/AAATscmyRBNw0FUmiqcbnqqLa/ARC091/E/in?preview=24.txt
・自力ならず。。テストケース見てバグ原因調べて、直ったらAC。
・構築系、LISとLDS
・案外、やろうとした方針自体は悪くなかった。
　考え方はソース内コメント及び自分ノート参照。
・結果的には、端数の扱いがよくなかった。切り上げてオーバーするところでbreakしてたのを、
　ちゃんと余り取って均等に割り振るようにしたら通った。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, A, B = MAP()

# 1はNとのペアしかできないので例外処理しておく
if A == 1 and B != N or B == 1 and A != N:
    print(-1)
    exit()
remain = N - A
# 作りたいグループ数(B-1)よりも、使える残数が少なかったらNG
if remain < B - 1:
    print(-1)
    exit()
# LIS以外の増加列を、何個ずつ、B-1箇所に割り振るか
d, m = divmod(remain, B-1) if B > 1 else (0, 0)
# 増加列をB個のグループに分けて、このグループ数をLDSにする
li = [[] for i in range(B)]
# LISにになる部分
li[0] = list(range(1, A+1))
j = A + 1
for i in range(1, B):
    tmp = []
    for _ in range(d):
        tmp.append(j)
        j += 1
    # 余りは各グループに1つずつ追加
    if m > 0:
        tmp.append(j)
        j += 1
        m -= 1
    # LISより長い列ができてしまったらNG
    if len(tmp) > A:
        print(-1)
        exit()
    li[i] = tmp
li = li[::-1]
ans = []
for grp in li:
    ans += grp
print(*ans)
