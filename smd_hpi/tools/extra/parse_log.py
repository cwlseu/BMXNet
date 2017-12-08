#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
parse mxnet output log into a markdown table
"""
import argparse
import sys
import re
import math
import random

import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
reload(sys)
sys.setdefaultencoding('utf-8') 


def set_param_update(epoch=79):
    x = [epoch]*2
    y = [0, 1]
    plt.plot(x, y, color="k", ls=':')

def main():
    parser = argparse.ArgumentParser(description='Parse mxnet output log')
    parser.add_argument('--logfile', type=str, default="train_ssd300_1.log",
                        help = 'the log file for parsing')
    parser.add_argument('--format', type=str, default='none',
                        choices = ['markdown', 'none'],
                        help = 'the format of the parsed outout')
    parser.add_argument('--targetfile', type=str, default="train_ssd300_bak.png",
                        help = 'the log file for parsing')
    args = parser.parse_args()
    train_info_parser(args)
    mAP_info_parser(args, 1)
    valid_info_parser(args, 1)

def valid_info_parser(args, step=10, verbose=False):
    class_names = "aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor".split(', ')
    
    assert len(class_names) == 20

    with open(args.logfile) as f:
        lines = f.readlines()

    r = re.compile('.*Epoch\[(\d+)\] Validation.*=([.\d]+)')

    data = {}
    i = 0
    for l in lines:
        m = r.match(l)
        if m is None:
            i = 0
            continue
        assert len(m.groups()) == 2
        epoch = int(m.groups()[0])
        if (epoch+1) % step != 0: continue
        val = float(m.groups()[1])

        if epoch not in data:
            data[epoch] = [0] * (len(class_names) + 1)

        data[epoch][i] = val
        i += 1

    if not verbose:
        if args.format == 'markdown':
            s = "| Epoch "
            for c in class_names:
                s  = "%s | %s" % (s, c)
            s  = "%s | %s |" % (s, "mAP")
            print s

            print "| --- "*(len(class_names)) + "|"

            for k, v in sorted(data.items(), key=lambda x: x[0]):
                s = "| %d " % (k+1)
                for i in v:
                    s+= "| %f " % (i)
                s +="|"
                print s

        elif args.format == 'none':
            s = "Epoch\t"
            for c in class_names:
                s  = "%s\t%s" % (s, c)
            s  = "%s\t%s" % (s, "mAP")
            print s
            for k, v in sorted(data.items(), key=lambda x: x[0]):
                s = "%d" % (k+1)
                for i in v:
                    s = "%s\t%f " % (s, i)
                print s
        elif args.format == 'latex':
            s = "Epoch &"
            for c in class_names:
                s  = "%s & %s" % (s, c)
            s  = "%s & %s\\" % (s, "mAP")

            for k, v in sorted(data.items(), key=lambda x: x[0]):
                s = "%d" % (k+1)
                for i in v:
                    s = "%s & %f" % (s, i)
                s += "\\"
                print s

    values = np.array(data.values())
    line_styles = ['-.', '-', ':', '--']

    # 绘制曲线
    for i in xrange(len(class_names)):
        cls_name = class_names[i]
        plt.plot(data.keys(), values[:, i], label=cls_name, linestyle=line_styles[random.randint(0, 3)])
    plt.plot(data.keys(), values[:, 20], label="mAP", color="red", linewidth=2.5)
    set_param_update(79)
    # 设置坐标镖头
    plt.xlabel('Train Epoch')
    plt.title('SSD_VGG16 training different class', fontsize=20)
    plt.grid(axis='y')
    # 设置坐标范围
    plt.ylim((0.0, 1.0))
    # 设置坐标显示的信息
    plt.xticks([y for y in range(0, 150, 10)])
    plt.yticks([float(i)/10 for i in xrange(1, 11)])
    # 图例显示
    plt.legend()
    # 存储显示图片
    plt.show()
    # plt.savefig(args.targetfile)


def train_info_parser(args, verbose=False):
    with open(args.logfile) as f:
        lines = f.readlines()

    res = [re.compile('.*Epoch\[(\d+)\] Train-CrossEntropy=([.\d]+)'),
           re.compile('.*Epoch\[(\d+)\] Train-SmoothL1=([.\d]+)'),
           re.compile('.*Epoch\[(\d+)\] Time.*=([.\d]+)'),
           re.compile('.*Epoch\[(\d+)\] Validation-mAP=([.\d]+)')]

    data = {}
    for l in lines:
        i = 0
        for r in res:
            m = r.match(l)
            if m is not None:
                break
            i += 1
        if m is None:
            continue

        assert len(m.groups()) == 2
        epoch = int(m.groups()[0])
        val = float(m.groups()[1])

        if epoch not in data:
            data[epoch] = [0] * len(res)

        data[epoch][i] = val

    if not verbose:
        if args.format == 'markdown':
            print "| Epoch | Train-CrossEntropy | Train-SmoothL1 | Time |"
            print "| --- | --- | --- | --- |"
            for k, v in data.items():
                print "| %2d | %f | %f | %.1f |" % (k+1, v[0], v[1], v[2])
        elif args.format == 'none':
            print "Epoch\tTrain-CrossEntropy\tTrain-SmoothL1\tTime"
            for k, v in data.items():
                print "%2d\t%f\t%f\t%.1f" % (k+1, v[0], v[1], v[2])
    values = np.array(data.values())

    # 绘制曲线
    plt.plot(data.keys(), values[:, 0], label="CrossEntropy", color="red", linewidth=1.5, linestyle='dashed')
    plt.plot(data.keys(), values[:, 1], label="SmoothL1 Loss", color="blue", linewidth=1.5, linestyle='dotted')
    plt.plot(data.keys(), values[:, 3], label="mAP", color="green", linewidth=1.5, linestyle='-.')
    set_param_update(epoch=79)
    # 设置坐标镖头
    plt.xlabel('Train Epoch')
    plt.title('SSD_VGG16 training information', fontsize=20)
    plt.grid(axis='y')
    # 设置坐标范围
    plt.ylim((0.0, 1.0))
    # 设置坐标显示的信息
    plt.xticks([y for y in range(0, 150, 10)])
    plt.yticks([float(i)/10 for i in xrange(1, 11)])
    # 图例显示
    plt.legend()
    # 存储显示图片
    plt.show()
    plt.savefig(args.targetfile)


def mAP_info_parser(args, step=1, target_file=None):
    with open(args.logfile) as f:
        lines = f.readlines()

    r = re.compile('.*Epoch\[(\d+)\] Validation-mAP=([.\d]+)')

    data = {}

    for l in lines:
        m = r.match(l)
        if m is None:
            continue
        assert len(m.groups()) == 2
        epoch = int(m.groups()[0])
        if (epoch+1) % step != 0: continue
        val = float(m.groups()[1])
        data[epoch+1] = val
    
    plt.plot(data.keys(), data.values(), label="mAP", color="green", linewidth=1.5, linestyle="-")
    set_param_update(epoch=79)
    plt.xlabel('Train Epoch')
    plt.ylabel('mAP')
    plt.ylim((0.2, 1.0))
    plt.grid(axis='y')
    plt.title('SSD_VGG16 mAP with training time', fontsize=24)
    plt.legend()
    if target_file is None:
        plt.show()
    else:
        plt.savefig(target_file)

if __name__ == '__main__':
    main()