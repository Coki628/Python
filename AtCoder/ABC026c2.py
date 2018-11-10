# -*- coding: utf-8 -*-

"""
自分の上司の最大部下給料と最小部下給料を更新しにいく版
- この方針だと、ループ1重で済むのでO(N)でいけると思う。
"""

class Worker:
    def __init__(self, sv):
        self.sv = sv
        self.salary = 0
        self.max_sub = 0
        self.min_sub = float('inf')

N = int(input())
workers = []
# 社長
workers.append(Worker(0))
# 他の社員
for i in range(N-1):
    workers.append(Worker(int(input())))

# 後ろから決めていけばうまくいく
workers.reverse()

for i in range(N):
    # 部下0人の調整
    if workers[i].min_sub == float('inf'):
        workers[i].min_sub = 0
    # 自分の給料確定
    workers[i].salary = workers[i].max_sub + workers[i].min_sub + 1
    # 自分の上司の最大部下給料と最小部下給料を更新しにいく(社長以外)
    if i != N-1:
        workers[N-workers[i].sv].max_sub = max(workers[N-workers[i].sv].max_sub, workers[i].salary)
        workers[N-workers[i].sv].min_sub = min(workers[N-workers[i].sv].min_sub, workers[i].salary)
# 配列の最後には社長の給料
print(workers[-1].salary)