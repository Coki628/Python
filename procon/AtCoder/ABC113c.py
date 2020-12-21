# -*- coding: utf-8 -*-

class City:
    def __init__(self, id, pref, year):
        self.id = id
        self.pref = pref
        self.year = year
        self.num = ''

N,M = map(int, input().split())
cities = []
for i in range(M):
    p, y = map(int, input().split())
    cities.append(City(i+1, p, y))

# year属性でソート
cities.sort(key=lambda city: city.year)

p_cnt = [0] * N
for i in range(M):
    # 0埋め
    pref_num = str(cities[i].pref).zfill(6)
    p_cnt[cities[i].pref-1] += 1
    city_num = str(p_cnt[cities[i].pref-1]).zfill(6)
    cities[i].num = pref_num + city_num

cities.sort(key=lambda city: city.id)

for i in range(M):
    print(cities[i].num)