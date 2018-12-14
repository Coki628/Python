# -*- coding: utf-8 -*-

"""
解説参考
・まず素因数分解で階乗全部バラす
・そこからいい感じの組み合わせを探して数える
"""

N = int(input())

# 階乗の素因数分解：ひとつずつ分解したものを合算
e = [0] * (N+1)
# 階乗の各値ループ
for i in range(2, N+1):
    num = i
    # 素因数分解するループ
    for j in range(2, N+1):
        while num % j == 0:
            num //= j
            e[j] += 1

# 約数75(各指数+1をかける)を取りうる素因数の組み合わせを探していく
nums = set()
# 指数が4, 4, 2
for p in range(2, N+1):
    for q in range(2, N+1):
        for r in range(2, N+1):
            if (e[p] >= 4 and e[q] >= 4 and e[r] >= 2
                    and p != q and q != r and r != p):
                nums.add(pow(p, 4) * pow(q, 4) * pow(r, 2))
# 指数が14, 4 or 24, 2
for p in range(2, N+1):
    for q in range(2, N+1):
        if e[p] >= 14 and e[q] >= 4 and p != q:
            nums.add(pow(p, 14) * pow(q, 4))
        if e[p] >= 24 and e[q] >= 2 and p != q:
            nums.add(pow(p, 24) * pow(q, 2))
# 指数が74
for p in range(2, N+1):
    if e[p] >= 74:
        nums.add(pow(p, 74))  

print(len(nums))
