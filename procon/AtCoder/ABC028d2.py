# -*- coding: utf-8 -*-

N,K = map(int, input().split())

# 中央値Kが何通りあるか
ans = 0
# 全部中央値
ans += 1
# 2つ中央値
ans += (N-1) * 3
# 1つ中央値
ans += (K-1) * (N-K) * 6
# 確率にするから全体で割る
print(ans / (N**3))