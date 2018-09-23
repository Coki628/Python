# -*- coding: utf-8 -*-

A, B, C = map(int, input().split())

# ようはどれか1つが10の桁になるだけ
print(max(A*10 + B + C, A + B*10 + C, A + B + C*10))