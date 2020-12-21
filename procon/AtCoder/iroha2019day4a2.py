# -*- coding: utf-8 -*-

"""
・300点自力AC
・とはいえ時間かかり過ぎ…。
・絶対もっとサクっとできるんだろうなぁと思いながらも
　頑張って条件分けまくった再帰探索。。枝刈り。
・で、解説見たけどWriterさんも300じゃないって言ってるししょうがないか。
・ちょっと落ち着いて、if文増やしまくったとこループにまとめた。(速くはなっていない。)
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N,A,B,C,D=MAP()
Al=LIST()+[0]
Bl=LIST()+[0]
Cl=LIST()+[0]
Dl=LIST()+[0]

# まずは明らかにダメなパターンを外す
if N*3!=sum([A,B,C,D]) or max(A,B,C,D)>N:
    No()
    exit()
# 1~3位の個数がおかしい
for i in range(1, 4):
    sm=Al.count(i)+Bl.count(i)+Cl.count(i)+Dl.count(i)
    if N!=sm:
        No()
        exit()

def rec(i, j, k, l, ans):
    # index外
    if i>A or j>B or k>C or l>D:
        return
    # 今回の4人の順位候補
    L=[Al[i], Bl[j], Cl[k], Dl[l]]
    # indexどこまで進んだか(0が最後)
    L2=[A-i, B-j, C-k, D-l]

    # 終了(失敗)条件、終了済が2人いる
    if L.count(0)>=2:
        return
    # 同順位候補が3人いる
    if L.count(1)==3 or L.count(2)==3 or L.count(3)==3:
        return
    # 同順位候補2人が2組いる
    for m in range(1, 4):
        for n in range(m+1, 4):
            if L.count(m)==2 and L.count(n)==2:
                return

    # 終了(成功)条件、3人あと1回で1人は終了済(今回4位)
    if L2.count(1)==3 and L2.count(0)==1:
        # 最後の4位を詰めて、結果出力
        ans+=str(L.index(0)+1)
        Yes()
        [print(s) for s in ans]
        exit()

    # 同順位候補2人のどちらかを4位とするパターンをそれぞれ試す
    L3=[i+1, j+1, k+1, l+1]
    for m in range(4):
        for n in range(m+1, 4):
            if L[m]==L[n]:
                # 4位にする人だけindexを進めない
                L3[m]-=1
                rec(L3[0], L3[1], L3[2], L3[3], ans+str(m+1))
                L3[m]+=1
                L3[n]-=1
                rec(L3[0], L3[1], L3[2], L3[3], ans+str(n+1))
                L3[n]+=1
    # 全員順位が確定的なパターン
    if L.count(1)==L.count(2)==L.count(3)==L.count(0)==1:
        for m in range(4):
            if L.index(0)==m:
                L3[m]-=1
                rec(L3[0], L3[1], L3[2], L3[3], ans+str(m+1))
                L3[m]+=1 
    return

rec(0, 0, 0, 0, '')
No()
