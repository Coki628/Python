# -*- coding: utf-8 -*-

""" 3次元DP """

N,A = map(int, input().split())
xN = list(map(int, input().split()))
X = max(max(xN), A)

# i(=xNリストからどこまでを選べる範囲とするか)からj(=0～N)枚選んで、選んだ値の合計をkにするような選び方の総数
dp = [[[0] * (N*X+1) for j in range(N+1)] for i in range(N+1)]
dp[0][0][0] = 1

ans = 0
for i in range(1,N+1):
    for j in range(N+1):
        for k in range(N*X+1):
            # 求めたい合計(k)が、この周回で増えた選択肢(xN[i-1])より小さい場合、
            # 選び方の総数は増える見込みがないので前回の値をそのまま使用
            if k < xN[i-1]:
                dp[i][j][k] = dp[i-1][j][k]
            # 合計kが大きければ、増やせるかもしれないので、選んだ数と合計を減らしたところを見に行く
            # 選べる数j=0は論外だし、添え字[j-1]に良くないので条件で除外する
            elif k >= xN[i-1] and j > 0:
                dp[i][j][k] = dp[i-1][j][k] + dp[i-1][j-1][k-xN[i-1]]
            # 選んだ数と合計から判断して、平均が条件に合うものなら答えに足す
            if i == N and j > 0 and k == j*A:
                ans += dp[i][j][k]

# ans = 0
# for k in range(1,N+1):
#     ans += dp[N][k][k*A]
        
print(ans)