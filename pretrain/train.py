# coding=utf-8
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np
from numpy import *
from collections import Counter
from operator import itemgetter
from parameters import Parameters


class Train(object):
    def __init__(self, parameters):
        self.parameters = parameters

    def train(self, data, single_model, single_optimizer, lr_single, criterion):
        total_loss = []
        sub_loss = []
        np.random.shuffle(data)
        i = 0
        while i < len(data)/self.parameters.batch_size:
            if (i % 200 == 0) and (i != 0):
                sub_avg_loss = np.mean(sub_loss, dtype=np.float64)
                print('==>Train sub_loss:{:0>2d} Loss:{:.4f}'.format(int(i/200), sub_avg_loss))
                sub_loss = []

            single_optimizer.zero_grad()

            j = 0
            pid_all = []
            target_all = []
            target_time_all = []
            uid_all = []
            tid_all = []
            category_all = []
            while (j < self.parameters.batch_size) and (self.parameters.batch_size*i+j < len(data)):
                record = data[self.parameters.batch_size*i+j]
                j += 1
                pid_all.append(record['present_pid'][0])
                if not self.parameters.is_baseline:
                    target_all.append(record['pid_label'][0])
                    target_time_all.append(record['tid_label'][0])
                    category_all.append(record['districts'][0])
                else:
                    target_all.append(record['pid_label'][0])
                    target_time_all.append(record['tid_label'][0])
                    category_all.append(record['present_districts'][0])
                uid_all.append(record['uids'][0])
                tid_all.append(record['present_tid'][0])
                

            i += 1
            pid_v = torch.LongTensor(pid_all)
            tid_v = torch.LongTensor(tid_all)
            target = torch.LongTensor(target_all)
            target_time_v = torch.LongTensor(target_time_all)
            uid_all_v = torch.LongTensor(uid_all)
            category_all = torch.LongTensor(category_all)

            if self.parameters.use_cuda:
                pid_v = pid_v.cuda()
                tid_v = tid_v.cuda()
                target = target.cuda()
                target_time_v = target_time_v.cuda()
                uid_all_v = uid_all_v.cuda()

            # target
            single_model.train()
            
            single_result = single_model(pid_v, uid_all_v, tid_v, target_time_v, category_all)
            target = target.view(-1)
            loss = criterion(single_result, target)

            loss.backward()

            # gradient clipping
            try:
                torch.nn.utils.clip_grad_norm_(
                    single_model.parameters(), self.parameters.clip)
                # prevent Exploding Gradients issue
            except:
                pass

            single_optimizer.step()
            sub_loss.append(loss.data.cpu().numpy())
            total_loss.append(loss.data.cpu().numpy())
        avg_loss = np.mean(total_loss, dtype=np.float64)  # compute avg loss

        return single_model, avg_loss


class Test(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.acc = np.zeros((3), dtype=np.float64)
        self.target_total = np.zeros((parameters.loc_size), dtype=np.float64)
        self.result_total = np.zeros((parameters.loc_size), dtype=np.float64)
        self.tp = np.zeros((parameters.loc_size), dtype=np.float64)
        self.recall = []
        self.precision = []
        self.cocin = 0
        self.l2sum = 0.0

    def test(self, data, single_model, criterion):
        i = 0
        self.acc = np.zeros((3), dtype=np.float64)
        self.cocin = 0
        self.l2sum = 0.0
        total_loss = []

        while i < len(data)/self.parameters.batch_size:
            j = 0
            pid = []
            target = []
            target_time = []
            uid = []
            tid = []
            category_all = []
            while (j < self.parameters.batch_size) and (self.parameters.batch_size*i+j < len(data)):
                record = data[self.parameters.batch_size*i+j]
                j += 1
                pid.append(record['present_pid'][0])
                if not self.parameters.is_baseline:
                    target.append(record['pid_label'][0])
                    target_time.append(record['tid_label'][0])
                    category_all.append(record['districts'][0])
                else:
                    target.append(record['pid_label'][0])
                    target_time.append(record['tid_label'][0])
                    category_all.append(record['present_districts'][0])
                uid.append(record['uids'][0])
                tid.append(record['present_tid'][0])

            i += 1

            pid_v = torch.LongTensor(pid)  # [1, window_size]
            tid_v = torch.LongTensor(tid)
            uid_v = torch.LongTensor(uid)
            target = torch.LongTensor(target)
            target_time_v = torch.LongTensor(target_time)
            category_all = torch.LongTensor(category_all)

            if self.parameters.use_cuda:
                pid_v = pid_v.cuda()
                tid_v = tid_v.cuda()
                uid_v = uid_v.cuda()
                target = target.cuda()
                target_time_v = target_time_v.cuda()

            # target
            single_model.eval()

            single_result = single_model(pid_v, uid_v, tid_v, target_time_v, category_all)
            target = target.view(-1)
            self.calculate_accuracy(target, single_result)
            loss = criterion(single_result, target)
            total_loss.append(loss.data.cpu().numpy())

        avg_loss = np.mean(total_loss, dtype=np.float64)  # compute avg loss
        for i in range(0, self.parameters.loc_size):
            if not self.target_total[i] == 0:
                self.recall.append(self.tp[i]/self.target_total[i])
            if not self.result_total[i] == 0:
                self.precision.append(self.tp[i]/self.result_total[i])
        recall_arr = np.array(self.recall)
        precision_arr = np.array(self.precision)
        m_recall = np.mean(recall_arr, dtype=np.float64)
        m_precision = np.mean(precision_arr, dtype=np.float64)
        if not (m_recall == 0 and m_precision == 0):
            F1_score = (2 * m_recall * m_precision) / (m_recall + m_precision)
        else:
            F1_score = 0
        print("macro-Recall:{:.4f} macro-Precision:{:.4f} F1-score:{:.4f}".format(
            m_recall, m_precision, F1_score))
        print("CoCiN:{:.4f} L2Dist:{:.4f}".format(
            self.cocin/(len(data)*self.parameters.target_num), 
            self.l2sum/(len(data)*self.parameters.target_num)))
        return self.acc / (len(data)*self.parameters.target_num), avg_loss

    def calculate_accuracy(self, target, scores):
        """target and scores are torch cuda Variable"""
        target = target.data.cpu().numpy()  # target=[pid,pid,pid,pid]
        _, idxx = scores.data.topk(10)  # val, idxx:[10]
        # Returns the k largest elements of the given input tensor along a given (the last) dimension.
        predx = idxx.cpu().numpy()

        for i, _ in enumerate(target):
            self.result_total[predx[i][0]] += 1  # for precision
            t = target[i]
            self.target_total[t] += 1  # for recall
            if t in predx[i][:10] and t > 0:  # top10
                self.acc[0] += 1
            if t in predx[i][:5] and t > 0:  # top5
                self.acc[1] += 1
            if t == predx[i][0] and t > 0:  # top1
                self.acc[2] += 1
                self.tp[t] += 1  # true sample, and divided into true samples
            if t != predx[i][0] and self.parameters.is_same_category(t, predx[i][0]):
                self.cocin += 1
            if t != predx[i][0]:
                self.l2sum += 0.2* self.parameters.l2_dist(t, predx[i][0])
