# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc031
・自力ならず。全然解法出なかった。
・まあ解説見て方針分かったら自力実装できたからまだいいか…。
・1~9の数字にabcを当てはめるのではなく、1~3の長さを当てはめて全探索する視点。
・確かにそれなら最大3^9だから十分全通り探索できる。
・練習のためにproduct使わないで再帰で自分でパターン網羅した版
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

K,N=MAP()
VW=[]
for i in range(N):
    v,w=input().split()
    VW.append((v, w))

# prodがvwに正しく当てはまるかを確認する
def check(prod, v, w):
    global ans
    # vを数値リストに分解
    vli=list(map(int, v))
    # 今回のprodでの、各文字の長さ(の終了index)
    slens=[prod[vli[0]-1]]
    for j in range(1, len(vli)):
        slens.append(prod[vli[j]-1]+slens[j-1])
    # slensの値に応じてwを切り取って検証用の文字列を作る
    stest=w[:slens[0]]
    # ansにwから切り取った文字列を当てはめる(この時に一致確認もする)
    if ans[vli[0]-1]=='' or ans[vli[0]-1]==w[:slens[0]]:
        ans[vli[0]-1]=w[:slens[0]]
    # 長さが合っても、一致しない文字列があればNG
    else: return False
    for j in range(1, len(slens)):
        # 長さがwより長くなったらNG
        if slens[j]>len(w): return False
        stest+=w[slens[j-1]:slens[j]]
        if ans[vli[j]-1]=='' or ans[vli[j]-1]==w[slens[j-1]:slens[j]]:
            ans[vli[j]-1]=w[slens[j-1]:slens[j]]
        else: return False
    # 検証文字列とwが一致したらOK、しなければNG
    return stest==w

# 全パターンの作成
prods=[]
def rec(li, cnt):
    if cnt==K:
        global prods
        prods.append(li)
        return
    rec(li+[1], cnt+1)
    rec(li+[2], cnt+1)
    rec(li+[3], cnt+1)
    return

rec([], 0)

# 1~Kの数字について1~3の長さを全通り試す
for prod in prods:
    # ansはprod毎に初期化
    ans=['']*K
    for i, vw in enumerate(VW):
        v,w=vw
        # 今回のprodがvwに正しく当てはまるかを全てのvwについて確認
        if not check(prod, v, w):
            break
    # 正しければ、結果出力
    else:
        [print(k) for k in ans]
        exit()
