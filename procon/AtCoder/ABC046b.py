# -*- coding: utf-8 -*-

N, K = map(int, input().split())

# 2個目以降のボールは組み合わせがひとつ減るのでK-1
print(K * ((K - 1) ** (N - 1)))