# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc135/editorial.pdf
　　　https://www.youtube.com/watch?v=brfeRxmzuKw
・数え上げDP
・桁DP系
・各桁独立で考えられる。(例：234 => 2*100+3*10+4*1)
・MOD13の世界なので、13か所のどこかに遷移させていく。
・添字にMODのどこなのかを使うDPはyahoo2017final_aでもあった。
・計算量は10万*13*10で1300万、pypyで0.7秒AC。
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

# やりやすさのため、小さい桁からいけるように逆順にしておく
S=input()[::-1]
N=len(S)

# dp[i][j] := 下からi桁を考えた時，13で割ったあまりがjであるものの数
dp=list2d(N+1, 13, 0)
dp[0][0]=1
mul=1
for i in range(N):
    for j in range(13):
        # ?だったら0~9を全て遷移させる
        if S[i]=='?':
            for k in range(10):
                # 桁を合わせる
                nxt=k*mul
                dp[i+1][(j+nxt)%13]+=dp[i][j]
                dp[i+1][(j+nxt)%13]%=MOD      
        else:
            # 決まっていたらその数字のみ遷移させる
            nxt=int(S[i])*mul
            dp[i+1][(j+nxt)%13]+=dp[i][j]
            dp[i+1][(j+nxt)%13]%=MOD
    # 桁合わせ用(10万桁とか直接使ったら死ぬので、毎回ここでMOD13する)
    mul*=10
    mul%=13
# 最終的にN桁終わって余り5の数が答え
print(dp[N][5])
