# -*- coding: utf-8 -*-

n,m = map(int, input().split())

# それぞれの位置を360度換算にする
# 短針に、長針で進む分も足す
n = n % 12 * 30 + m / 12 * 6
m = m * 6

print(abs(n - m))