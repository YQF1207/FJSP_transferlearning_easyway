import numpy as np
import random
from collections import defaultdict
import copy
#from Instance_8 import Processing_time
from dfjs01 import  Processing_time


processing_time = Processing_time
#job_num = len(processing_time)
#print(job_num)
zero_model = copy.deepcopy(processing_time)
job_num = 10
fac_num = 3
job_num_count = 0
job_in_fac = []
def split_interger(job_num, fac_num):
    assert  fac_num>0
    zhenchu = job_num // fac_num
    yushu = job_num % fac_num
    if yushu > 0:
        return [zhenchu]*(fac_num-yushu)+[zhenchu+1]*yushu
    if yushu < 0 :
        return [zhenchu-1]*-yushu + [zhenchu]*(fac_num+yushu)
    return [zhenchu]*fac_num

a = split_interger(job_num,fac_num)
for i in range(len(a)):
    a[i]+=1
#print(a)

sub_factory = defaultdict(list)
#print(processing_time)
begin = 0
for i in range(fac_num):  #多个车间都是0
    start, end = [begin,begin+a[i]]
    #print(start, end)
    begin = begin+a[i]
    for key in range(start, end-1):
        #print(key)
        sub_factory['factory_' + str(i)].append(processing_time[key-i])
    sub_factory['factory_' + str(i)].append((np.zeros(np.shape(processing_time[0]))).tolist())
    #print(sub_factory['factory_' + str(i)])


def find_useless_job(fac):  #单个车间工件为0的下标
    b = []
    for i in range(len(fac)):
        if np.all(fac[i]) == 0:
            b.append(i)
    return b
fac = [6,0,9]


a = [0,1,2,3]
print(a[:2])
#print(sorted(fac))

#print(sub_factory)
#job = []

# k =[[[7, 9999, 4], [8, 3, 9999], [3, 9999, 6], [2, 4, 9999]],
# [[8, 12, 9999], [9999, 14, 9999], [7, 14, 4], [8, 9999, 4]],
# [[10, 15, 8], [9999, 2, 6], [2, 9999, 4], [6, 3, 9999]],
# [[9999, 9, 5], [6, 9999, 2], [9999, 7, 12], [9, 6, 3]],
# [[10, 9999, 15], [9999, 7, 14], [5, 8, 9999], [4, 6, 8]],
# [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
# ]
#
# b = []
# for i in range(len(k)):
#     if np.all(k[i]) == 0:
#         b.append(i)
# print(b)

# 交换
# for i in range(fac_num):  # 多个车间都是0
#     fac = list(sub_factory['factory_' + str(i)])
#     print(fac)
# a = 0
# b = 0
# exchange = sub_factory['factory_0'][a]
# sub_factory['factory_0'][a] = sub_factory['factory_1'][b]
# sub_factory['factory_1'][b] = exchange
# print('......')
# for i in range(fac_num):  # 多个车间都是0
#     fac = list(sub_factory['factory_' + str(i)])
#     print(fac)




# #shuffle = true or false
#
# # def reset_to_0(array):
# #     for i, e in enumerate(array):
# #             if isinstance(e, list):
# #                 reset_to_0(e)
# #             else:
# #                 array[i] = 0
# #
# # reset_to_0(zero_model)
#
#
# #print(zero_model)
# job = random.randint(0,job_num)   #删除工件 用0代替
# print(job)
# for i in range(len(processing_time)-1):
#     if i == job:
#         print(i)
#         # print(processing_time[i])
#         print(np.shape(processing_time[i]))
#         processing_time[i] = np.zeros(np.shape(processing_time[i])).tolist()
# print(processing_time)
#
# factory_num = 3
# #np.random.shuffle(processing_time)
# # for i in range(factory_num):
# #     globals()['factory'+str(i)]=zero_model   #多个车间都是0
#
#
#
#
# rank = []
# for i in range(len(processing_time)):
#     rank.append(i)
# origin_dict = dict(zip(rank,processing_time)) #工件和序号绑定的字典
# print(origin_dict)
#
#
#
# sub_factory = defaultdict(list)
# for i in range(factory_num):  #多个车间都是0
#     sub_factory['factory_'+str(i)].extend(zero_model)
#     sub_factory['factory_' + str(i)] = dict(zip(rank,sub_factory['factory_' + str(i)]))
# #print(sub_factory['factory_0'])
#
#
# job_in_factory = len(processing_time)//factory_num
# num_in_each_factor =[]
# for i in range(factory_num):
#     start,end = i*job_in_factory,(i+1)*job_in_factory
#     for key in range(start,end):
#         sub_factory['factory_' + str(i)][key] = origin_dict[key]
#
#     num = end - start
#     if factory_num * job_in_factory != len(processing_time) and i == (factory_num-1):
#         for j in (end,len(processing_time)-1):
#             sub_factory['factory_' + str(factory_num-1)][j] = origin_dict[j]
#         num = num + (len(processing_time) - factory_num*job_in_factory)
#     num_in_each_factor.append(num)
#
# print(sub_factory['factory_0'])
# print(sub_factory['factory_1'])
# print(sub_factory['factory_2'])
# print(num_in_each_factor)

#
# def processing_time_data( processing_time):
#     J_num = len(processing_time)
#     machine_num = 0
#     o_num = 0
#     o = []
#     n = []
#     for k, v in enumerate(processing_time):
#         o_num += len(v)
#         o.append(len(v))
#         n.append(k + 1)
#     J = dict(zip(n, o))
#     for i in range(len(processing_time[0])):
#         machine_num = len(processing_time[0][i])
#     return  o_num
#
# print(processing_time_data(processing_time))




