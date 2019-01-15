# -*- coding: utf-8 -*-
 
"""
参考：https://img.atcoder.jp/abc075/editorial.pdf
・自力ACしたのがO(N^4)でこれはO(N^5)の愚直解
・制約からして何とかなりそうだし、こっちのが実装全然軽いし実戦向きぽいね。
　→O(N^5)=50^5=3億1250万はpypyでも無理だったわ。
"""
 
N, K = map(int, input().split())
 
X, Y = [0]*N, [0]*N
for i in range(N):
    X[i], Y[i] = map(int, input().split())
 
X2 = sorted(set(X))
Y2 = sorted(set(Y))
 
ans = float('inf')
# 可能性のある全長方形ループ
for x1 in X2:
    for y1 in Y2:
        for x2 in X2:
            for y2 in Y2:
                cnt = 0
                # 各頂点ループで範囲内にあるか1つずつ確認する
                for i in range(N):
                    if x1 <= X[i] <= x2 and y1 <= Y[i] <= y2:
                        cnt += 1
                if cnt >= K:
                    ans = min((x2 - x1) * (y2 - y1), ans)
print(ans)
