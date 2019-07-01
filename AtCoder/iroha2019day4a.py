# -*- coding: utf-8 -*-

"""
・300点自力AC
・とはいえ時間かかり過ぎ…。
・絶対もっとサクっとできるんだろうなぁと思いながらも
　頑張って条件分けまくった再帰探索。。枝刈り。
・で、解説見たけどWriterさんも300じゃないって言ってるししょうがないか。
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
sm1=Al.count(1)+Bl.count(1)+Cl.count(1)+Dl.count(1)
sm2=Al.count(2)+Bl.count(2)+Cl.count(2)+Dl.count(2)
sm3=Al.count(3)+Bl.count(3)+Cl.count(3)+Dl.count(3)
if not (N==sm1==sm2==sm3):
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

    # 終了(成功)条件、3人あと1回で1人は終了済(今回4位)
    if L2.count(1)==3 and L2.count(0)==1:
        # 最後の4位を詰めて、結果出力
        ans+=str(L.index(0)+1)
        Yes()
        [print(s) for s in ans]
        exit()

    # 終了(失敗)条件、終了済が2人いる
    if L.count(0)>=2:
        return
    # 同順位候補が3人いる
    if L.count(1)==3 or L.count(2)==3 or L.count(3)==3:
        return
    # 同順位候補2人が2組いる
    if (L.count(1)==2 and L.count(2)==2 or L.count(1)==2 and L.count(3)==2
            or L.count(2)==2 and L.count(3)==2):
        return

    # 同順位候補2人のどちらかを4位とするパターンをそれぞれ試す
    if L[0]==L[1]:
        # 4位にする人だけindexを進めない
        rec(i, j+1, k+1, l+1, ans+'1')
        rec(i+1, j, k+1, l+1, ans+'2')
    elif L[0]==L[2]:
        rec(i, j+1, k+1, l+1, ans+'1')
        rec(i+1, j+1, k, l+1, ans+'3')
    elif L[0]==L[3]:
        rec(i, j+1, k+1, l+1, ans+'1')
        rec(i+1, j+1, k+1, l, ans+'4')
    elif L[1]==L[2]:
        rec(i+1, j, k+1, l+1, ans+'2')
        rec(i+1, j+1, k, l+1, ans+'3')
    elif L[1]==L[3]:
        rec(i+1, j, k+1, l+1, ans+'2')
        rec(i+1, j+1, k+1, l, ans+'4')
    elif L[2]==L[3]:
        rec(i+1, j+1, k, l+1, ans+'3')
        rec(i+1, j+1, k+1, l, ans+'4')
    # 全員順位が確定的なパターン
    elif L.count(1)==L.count(2)==L.count(3)==L.count(0)==1:
        if L.index(0)==0:
            rec(i, j+1, k+1, l+1, ans+'1')
        elif L.index(0)==1:
            rec(i+1, j, k+1, l+1, ans+'2')
        elif L.index(0)==2:
            rec(i+1, j+1, k, l+1, ans+'3')
        else:
            rec(i+1, j+1, k+1, l, ans+'4')
    return

rec(0, 0, 0, 0, '')
No()
