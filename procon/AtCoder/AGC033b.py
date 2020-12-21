# -*- coding: utf-8 -*-

"""
・600点初自力AC！
・しかも参戦中に、これは嬉しい！
・と思ったら、この貪欲は嘘解法って話が…。
・とはいえC++の提出早い人の見ても結構この方針でやってるぽいのだが、どうなんだろうか…。
・でも確かにこの貪欲で解けちゃうなら600点には易し過ぎる気もするしなー。。
・嘘解法確認。以下ケースは本当はNOだけど、YESになる。
2 2 3
1 1
RDU
DRR
・TODO: まあそのうちちゃんとしたやつやろう。。
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

H,W,N=MAP()
sr,sc=MAP()
S=input()
T=input()

h=sr
for i in range(N):
    if S[i]=='U':
        h-=1
    if h==0:
        NO()
        exit()
    if T[i]=='D':
        if h<H:
            h+=1
h=sr
for i in range(N):
    if S[i]=='D':
        h+=1
    if h>H:
        NO()
        exit()
    if T[i]=='U':
        if h>1:
            h-=1
w=sc
for i in range(N):
    if S[i]=='L':
        w-=1
    if w==0:
        NO()
        exit()
    if T[i]=='R':
        if w<W:
            w+=1
w=sc
for i in range(N):
    if S[i]=='R':
        w+=1
    if w>W:
        NO()
        exit()
    if T[i]=='L':
        if w>1:
            w-=1
YES()
