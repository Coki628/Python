# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc075/editorial.pdf
・自力ACしたのがO(N^4)でこれはO(N^5)の愚直解
・制約からして何とかなりそうだし、こっちのが実装全然軽いし実戦向きぽいね。
　→O(N^5)=50^5=3億1250万はpypyでも無理だったわ。
・添字見づらくなったけど定数倍改善。多分半分で約1憶5000万かと。
　→無事1憶5000万もpypyならAC
"""

N, K = map(int, input().split())

X, Y = [0]*N, [0]*N
for i in range(N):
    X[i], Y[i] = map(int, input().split())

X2 = sorted(set(X))
Y2 = sorted(set(Y))

ans = float('inf')
# 可能性のある全長方形ループ
for i in range(len(X2)):
    for j in range(len(Y2)):
        for k in range(i+1, len(X2)):
            for l in range(j+1, len(Y2)):
                cnt = 0
                # 各頂点ループで範囲内にあるか1つずつ確認する
                for m in range(N):
                    if X2[i] <= X[m] <= X2[k] and Y2[j] <= Y[m] <= Y2[l]:
                        cnt += 1
                if cnt >= K:
                    ans = min((X2[k] - X2[i]) * (Y2[l] - Y2[j]), ans)
print(ans)
