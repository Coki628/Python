# -*- coding: utf-8 -*-

X = int(input())
A = int(input())
B = int(input())

# ケーキを1個
X -= A

# ドーナツを買った余り
print(X % B)