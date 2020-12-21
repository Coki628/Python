# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc096/editorial.pdf
・こういうの、自分で思いつくようになりたい。
→同じ回数(今回なら5回)足すのであればmodを揃えればその倍数にコントロールできる。
"""

# 素数列挙(エラトステネスの篩)
def eratosthenes_sieve(n):
    table = [0] * (n + 1)
    prime_list = []
    
    for i in range(2, n + 1):
        if table[i] == 0:

            # mod 5が同じ数なら5個集まれば5の倍数になるから合成数
            if i % 5 == 1:
	            prime_list.append(i)
            
            for j in range(i + i, n + 1, i):
                table[j] = 1
    
    return prime_list

N = int(input())
print(*(eratosthenes_sieve(10000)[:N]))

