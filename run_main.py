import numpy as np
#from processing_time import Processing_time
from collections import defaultdict
import copy
from la18 import Processing_time
import random
import main
import matplotlib.pyplot as plt
import main2
import main_contrast
import time

class run_main:
    def __init__(self, processing_time, factory_number, job_shuffle):
        self.processing_time = processing_time
        self.factory_number = factory_number
        self.job_shuffle = job_shuffle
        self.job_num = len(processing_time)
        # #print(self.job_shuffle)
        # if self.job_shuffle:
        #     processing_time = random.shuffle(processing_time)



    def plot_fitness_random(self, value1, factory, iteration):
      x = np.linspace(0, 30, 30)
      plt.plot(x, value1, 'b:')
      #plt.plot(x, value2, '-k')

      plt.title(
          'the maximum completion time of each iteration')
      plt.ylabel('Cmax')
      plt.xlabel('Iteration num')
      plt.savefig('./plot/'+'transfer_iteration_'+str(factory)+'_'+str(iteration))
      plt.close()

    def split_interger(self,job_num, fac_num):
        assert fac_num > 0
        zhenchu = job_num // fac_num
        yushu = job_num % fac_num
        if yushu > 0:
            return [zhenchu] * (fac_num - yushu) + [zhenchu + 1] * yushu
        if yushu < 0:
            return [zhenchu - 1] * -yushu + [zhenchu] * (fac_num + yushu)
        return [zhenchu] * fac_num

    def assign(self, fac_num, each_fac_num):
        sub_factory = defaultdict(list)
        #print(processing_time)
        #null = [9999,0,9999],[9999,0,9999],[9999,0,9999],[9999,0,9999]  #可能会减少gantte画错，但还是有问题
        #null = [9999,9999,0,9999,9999],[9999,9999,0,9999,9999],[9999,9999,0,9999,9999],[9999,9999,0,9999,9999],[9999,9999,0,9999,9999]
        #null = [9999, 9999, 0, 9999, 9999], [9999, 9999, 0, 9999, 9999], [9999, 9999, 0, 9999, 9999], [9999, 9999, 0, 9999, 9999], [9999, 9999, 0, 9999, 9999], [9999, 9999, 0, 9999, 9999]
        null =[9999, 9999, 9999, 0 ,9999, 9999, 9999, 9999, 9999, 9999],[9999, 9999, 9999, 9999, 0, 9999, 9999, 9999, 9999, 9999],[9999, 9999, 9999, 9999, 0, 9999, 9999, 9999, 9999, 9999],[9999, 9999, 9999, 9999, 0, 9999, 9999, 9999, 9999, 9999],[9999, 9999, 9999, 9999, 0, 9999, 9999, 9999, 9999, 9999],[9999, 9999, 9999, 9999, 0, 9999, 9999, 9999, 9999, 9999],[9999, 9999, 9999, 9999, 0, 9999, 9999, 9999, 9999, 9999],[9999, 9999, 9999, 9999, 0, 9999, 9999, 9999, 9999, 9999],[9999, 9999, 9999, 9999, 0, 9999, 9999, 9999, 9999, 9999],[9999, 9999, 9999, 9999, 0, 9999, 9999, 9999, 9999, 9999]
        begin = 0
        origin_assignment = []
        for i in range(fac_num):  # 多个车间都是0
            assignment = []
            start, end = [begin, begin + each_fac_num[i]]
            #print(start, end)
            begin = begin + each_fac_num[i]
            for key in range(start, end - 1):
                sub_factory['factory_' + str(i)].append(processing_time[key-i])
                assignment.append(key)
            #sub_factory['factory_' + str(i)].append((np.zeros(np.shape(processing_time[0]))).tolist())

            #暂时注释
            sub_factory['factory_' + str(i)].append(list(null))
            assignment.append(key+1)
            origin_assignment.append(assignment)
            #print(sub_factory)
        return sub_factory, origin_assignment



    def processing_time_data(self, processing_time):
        J_num = len(processing_time)
        machine_num = 0
        o_num = 0
        o = []
        n = []
        for k, v in enumerate(processing_time):
            o_num += len(v)
            o.append(len(v))
            n.append(k + 1)
        J = dict(zip(n, o))
        for i in range(len(processing_time[0])):
            machine_num = len(processing_time[0][i])
        return J, machine_num, J_num, o_num

    def find_useless_job(self,fac):  #单个车间工件为0的下标
        b = []
        for i in range(len(fac)):
            if np.all(fac[i]) == 0:
                b.append(i)
        return b


    def exchange_jobs(self,fac1_max_index, fac2_min_index, job_max, job_min,sub_fact,process_assignment, transfer_max_index,transfer_min_index,C_max,process_assignment_best,useless_all,processed_all):
        print('/////////')
        print(process_assignment[fac1_max_index])
        print(process_assignment[fac2_min_index])
        #print(C_max)
        print('/////////')
        #print(transfer)
        #exchange_main = main2.GA_exchange()
        sub_fact_origin = copy.deepcopy(sub_fact)
        process_assignment_origin = copy.deepcopy(process_assignment)

        transfer_max = transfer_max_index
        transfer_min = transfer_min_index

        # useless_max = useless_all[fac1_max_index]
        # useless_min = useless_all[fac2_min_index]
        #processed_all = []

        #print(useless_min)

        exchange_best = []
        exchange_process_assignment = []
        exchange_sub_fact = []

        exchange_worst = []
        exchange_worst_all = []
        exchange_worst_process_assignment = []
        exchange_worst_sub_fact = []


        #a = np.random.randint(0,len(process_assignment[fac1_max_index]))
        #b = np.random.randint(0,len(process_assignment[fac2_min_index]))
        #print(a,b)
        for job_a in range(job_max):
            for job_b in range(job_min):
                try:
                    print('尝试交换')
                    process_assignment = copy.deepcopy(process_assignment_origin)
                    print('runnning:::',process_assignment)

                    sub_fact = copy.deepcopy(sub_fact_origin)
                    c_max = []

                    exchange_job = process_assignment[fac1_max_index][job_a]
                    process_assignment[fac1_max_index][job_a] = process_assignment[fac2_min_index][job_b]
                    process_assignment[fac2_min_index][job_b] = exchange_job
                    print('runnning_exchanged:::',process_assignment)


                    exchange_job1 = 0
                    exchange_job2 = 0
                    for i in range(len(sub_fact['factory_' + str(fac1_max_index)])):
                        if i == job_a:
                            exchange_job1 = sub_fact['factory_' + str(fac1_max_index)][i]

                    for j in range(len(sub_fact['factory_' + str(fac2_min_index)])):
                        if j == job_b:
                            exchange_job2 = sub_fact['factory_' + str(fac2_min_index)][j]
                            sub_fact['factory_' + str(fac2_min_index)][j] = exchange_job1

                    for k in range(len(sub_fact['factory_' + str(fac1_max_index)])):
                        if k == job_a:
                            sub_fact['factory_' + str(fac1_max_index)][k] = exchange_job2

                    if copy.deepcopy(sorted(process_assignment[fac1_max_index])) in processed_all:
                        print('                                               已经存在')
                        continue
                    g2 = main2.GA_exchange()

                    J, machine_num, J_num, o_num = run.processing_time_data(sub_fact['factory_' + str(fac1_max_index)])
                    #useless_job_exchange = run.find_useless_job(sub_fact['factory_' + str(fac1_max_index)])  # 判断全为0的工件
                    #print("useless:", useless_job_exchange)
                    best_max, best_each, transfer_select_chs = g2.main(sub_fact['factory_' + str(fac1_max_index)], J, machine_num, J_num, o_num, job_a, job_b, transfer_max,process_assignment[fac1_max_index],useless_job,job_a,job_b,1,False)
                    print('----')
                    J, machine_num, J_num, o_num = run.processing_time_data(sub_fact['factory_' + str(fac2_min_index)])
                    #useless_job_exchange = run.find_useless_job(sub_fact['factory_' + str(fac2_min_index)])  # 判断全为0的工件
                    #print("useless:", useless_job_exchange)
                    best_min, best_each, transfer_select_chs = g2.main(sub_fact['factory_' + str(fac2_min_index)], J, machine_num, J_num, o_num, job_a, job_b, transfer_min,process_assignment[fac2_min_index],useless_job,job_a,job_b,2,False)
                    c_max.append(min(best_max))
                    c_max.append(min(best_min))
                    c_max.append(fac1_max_index)
                    c_max.append(fac2_min_index)
                    print('...................')
                    print(job_a,job_b)
                    print('...................')
                    #print(sub_fact)
                    print(c_max)


                    if max(c_max[:2]) <= C_max:
                        print('                                     appear smaller')
                        exchange_best.append(max(c_max[:2]))
                        process_assignment_best.append(process_assignment) ###把所有出现过小于cmax的组合跳过

                        exchange_process_assignment.append(process_assignment)
                        exchange_sub_fact.append(sub_fact)

                    #elif min(c_max) > C_max:
                    elif max(c_max[:2])-C_max>=20 and C_max-min(c_max[:2])>=20 :
                        print('                                     much bigger')
                        exchange_worst_all.append(c_max)
                        #exchange_worst.append(min(c_max))                ##最小值大于cmax的进行转移
                        exchange_worst_process_assignment.append(process_assignment)
                        exchange_worst_sub_fact.append(sub_fact)

                        processed_all.append(copy.deepcopy(sorted(process_assignment[fac1_max_index])))
                        processed_all.append(copy.deepcopy(sorted(process_assignment[fac2_min_index])))
                    else:
                        processed_all.append(copy.deepcopy(sorted(process_assignment[fac1_max_index])))
                        processed_all.append(copy.deepcopy(sorted(process_assignment[fac2_min_index])))
                except:
                    print("something wrong")
                    continue
##############################################
        print('尝试移动')
        if exchange_worst_all != []:
            for worst in range(len(exchange_worst_all)):
                print(exchange_worst_all[worst])

                worse_min = min((exchange_worst_all[worst])[:2])
                #print(worse_min)
                worse_min_index = exchange_worst_all[worst][:2].index(worse_min)
                #print(worse_min_index)
                worst_min_index = exchange_worst_all[worst][worse_min_index+2]
                print(worst_min_index)

                worse_max = max((exchange_worst_all[worst])[:2])
                worse_max_index = exchange_worst_all[worst][:2].index(worse_max)

                worst_max_index = exchange_worst_all[worst][worse_max_index+2]
                print(worst_max_index)

                #process_worst_assignment_min = exchange_worst_process_assignment[worst][worst_min_index]
                #process_worst_assignment_max = exchange_worst_process_assignment[worst][worst_max_index]
                #print('==================================')
                #print(exchange_worst_sub_fact[worst])
                # print(exchange_worst_sub_fact[worst][worst_max_index])
                # print(exchange_worst_sub_fact[worst][worst_min_index])
                #print('==================================')

                #run.find_useless_job(sub_fact['factory_' + str(fac2_min_index)])

                useless_in_process_worst_min = run.find_useless_job(exchange_worst_sub_fact[worst]['factory_' + str(worst_min_index)])  # 判断全为0的工件
                useless_in_process_worst_max = run.find_useless_job(exchange_worst_sub_fact[worst]['factory_' + str(worst_max_index)])  # 判断全为0的工件
                exchange_worst_sub_fact_once = exchange_worst_sub_fact[worst]

                print(exchange_worst_process_assignment[worst])

                if useless_in_process_worst_min == []:
                    continue
                else:
                    for job in range(len(exchange_worst_process_assignment[worst][worst_max_index])):
                        for unuse in useless_in_process_worst_min:
                            try:
                                sub_fact = copy.deepcopy(exchange_worst_sub_fact_once)
                                process_assignment = copy.deepcopy(exchange_worst_process_assignment[worst])
                                print('unuse:',unuse)
                                print('runnning_exchanged_before:::',process_assignment)
                                c_max = []
                                exchange_job = process_assignment[worst_max_index][job]
                                process_assignment[worst_max_index][job] = process_assignment[worst_min_index][unuse]
                                process_assignment[worst_min_index][unuse] = exchange_job
                                print('runnning_move:::', process_assignment)

                                exchange_job1 = 0
                                exchange_job2 = 0
                                for i in range(len(sub_fact['factory_' + str(worst_max_index)])):
                                    if i == job:
                                        exchange_job1 = sub_fact['factory_' + str(worst_max_index)][i]

                                for j in range(len(sub_fact['factory_' + str(worst_min_index)])):
                                    if j == unuse:
                                        exchange_job2 = sub_fact['factory_' + str(worst_min_index)][j]
                                        sub_fact['factory_' + str(worst_min_index)][j] = exchange_job1

                                for k in range(len(sub_fact['factory_' + str(worst_max_index)])):
                                    if k == job:
                                        sub_fact['factory_' + str(worst_max_index)][k] = exchange_job2

                                if copy.deepcopy(sorted(process_assignment[worst_max_index])) in processed_all:
                                    print('                                               已经存在')
                                    continue
                                g2 = main2.GA_exchange()

                                J, machine_num, J_num, o_num = run.processing_time_data(sub_fact['factory_' + str(worst_max_index)])

                                best_max, best_each, transfer_select_chs = g2.main(sub_fact['factory_' + str(worst_max_index)], J, machine_num,
                                                                                   J_num, o_num, job, unuse, transfer_max,
                                                                                   process_assignment[worst_max_index], useless_job, job,
                                                                                   unuse, 1, True)

                                J, machine_num, J_num, o_num = run.processing_time_data(sub_fact['factory_' + str(worst_min_index)])
                                #useless_job_exchange = run.find_useless_job(sub_fact['factory_' + str(worst_min_index)])  # 判断全为0的工件
                                #print("useless_exchange:", useless_job_exchange)
                                print('----')
                                best_min, best_each, transfer_select_chs = g2.main(sub_fact['factory_' + str(worst_min_index)], J, machine_num,
                                                                                   J_num, o_num, job, unuse, transfer_min,
                                                                                   process_assignment[worst_min_index], useless_job, job,
                                                                                   unuse, 2, True)

                                processed_all.append(copy.deepcopy(sorted(process_assignment[worst_max_index])))
                                processed_all.append(copy.deepcopy(sorted(process_assignment[worst_min_index])))

                                c_max.append(min(best_max))
                                c_max.append(min(best_min))
                                if max(c_max) <= C_max:
                                    print('                                     appear smaller')
                                    process_assignment_best.append(process_assignment)
                                    exchange_best.append(max(c_max))
                                    exchange_sub_fact.append(sub_fact)
                                    exchange_process_assignment.append(process_assignment)
                            except :
                                print('something wrong ')
                            continue

###############################################


        if exchange_best == []:
            a = np.random.randint(0,len(process_assignment[fac1_max_index]))
            b = np.random.randint(0,len(process_assignment[fac2_min_index]))
            exchange_job1 = 0
            exchange_job2 = 0

            exchange_job = process_assignment[fac1_max_index][a]
            process_assignment[fac1_max_index][a] = process_assignment[fac2_min_index][b]
            process_assignment[fac2_min_index][b] = exchange_job

            for i in range(len(sub_fact['factory_' + str(fac1_max_index)])):
                if i == a:
                    exchange_job1 = sub_fact['factory_' + str(fac1_max_index)][i]

            for j in range(len(sub_fact['factory_' + str(fac2_min_index)])):
                if j == b:
                    exchange_job2 = sub_fact['factory_' + str(fac2_min_index)][j]
                    sub_fact['factory_' + str(fac2_min_index)][j] = exchange_job1

            for k in range(len(sub_fact['factory_' + str(fac1_max_index)])):
                if k == a:
                    sub_fact['factory_' + str(fac1_max_index)][k] = exchange_job2

            return sub_fact,process_assignment

        else:
            print(min(exchange_best))
            exchange_min_index = exchange_best.index(min(exchange_best))
            print('exchange_min_index',exchange_min_index)
            #print(exchange_sub_fact[exchange_min_index])
            print(exchange_process_assignment[exchange_min_index])
            processed_all.append(copy.deepcopy(sorted(exchange_process_assignment[exchange_min_index][0])))
            processed_all.append(copy.deepcopy(sorted(exchange_process_assignment[exchange_min_index][1])))
            return exchange_sub_fact[exchange_min_index], exchange_process_assignment[exchange_min_index]


#exchange中计算出最优解，到循环中会出现不同的结果
if __name__=='__main__':


    job_shuffle = False
    processing_time = Processing_time
    if job_shuffle:
        np.random.shuffle(processing_time)
    factory_num = 2
    job_num = 10
    Max_iteration = 15

    transfer = []
    for i in range(factory_num):
        transfer.append(0)

    processed_all = []

    run = run_main(processing_time, factory_num, job_shuffle)
    each_fac_num = run.split_interger(job_num,factory_num)

    for i in range(len(each_fac_num)):
        each_fac_num[i] += 1

    #print(each_fac_num)

    #print(each_fac_num)
    sub_fact, process_assignment = run.assign(factory_num,each_fac_num)
    #print(sub_fact)
    print(process_assignment)       #工件实际位置     d.gantte是索引
    makespan = []
    process_assignment_best = []
    best_result = []
    #all_process_assignment = []

    for iteration in range(Max_iteration):
        print('iteration:',iteration)
        times = []
        Job = []
        useless_all = []
        try:
            for i in range(factory_num):
                print("factory:",i)
                fac = list(sub_fact['factory_' + str(i)])
                useless_job = run.find_useless_job(fac) #判断全为0的工件
                print("useless_main:",useless_job)
                J, machine_num, J_num, o_num = run.processing_time_data(fac)
                #print(J, machine_num, J_num, o_num)
                g = main.GA()
                print(process_assignment[i])
                best, best_each, transfer_select_chs = g.main(fac, J, machine_num, J_num, o_num, iteration, i, transfer[i],process_assignment[i],useless_job)
                transfer[i] = transfer_select_chs

                #k = main_contrast.GA_contrast()
                #best_contrast, best_each_contrast = k.main_contrast(fac,J,machine_num,J_num,o_num)
                useless_all.append(useless_job)
                times.append(min(best))
                Job.append(J_num)
                run.plot_fitness_random(best,i,iteration)
            # print('车间工件数')
            print(Job)#工件加一后
            print(times)
            best_result.append(times)
            rank_time = sorted(copy.deepcopy(times))
            min1 = rank_time[0]  #找到最小两个fit
            min1_index = times.index(min1)
            min_job = Job[min1_index]
            max1 = rank_time[-1] #找到最大两个fit
            max1_index = times.index(max1)
            max_job = Job[max1_index]

            if min1 == max1:
                max1_index += 1
            print("交换车间")
            print(min1_index)
            print(max1_index)

            print("交换车间的工件数量")
            print(min_job)
            print(max_job)

            print(max1)

            print(process_assignment)
            #print(transfer)
            sub_fact ,process_assignment  = run.exchange_jobs(max1_index, min1_index, max_job, min_job,sub_fact,process_assignment,transfer[max1_index],transfer[min1_index],max1,
                                                              process_assignment_best, useless_all,processed_all)
            process_assignment_best.append(process_assignment)
            print(process_assignment)

            makespan.append(max1)
            print('......................')
        except:
            continue
    print(best_result)

    x = np.linspace(0, Max_iteration, Max_iteration)
    plt.plot(x, makespan, 'r:')
      #plt.plot(x, value2, '-b')

    plt.title(
          'the maximum makespan')
    plt.ylabel('Cmax')
    plt.xlabel('Iteration num')
    plt.savefig('makespan')
    plt.close()


