#coding=utf-8
"""
为精分类模型Alexnet制造训练和测试数据，使用yolov3检测的结果
"""
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os.path as osp
from darknet import Darknet


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module Make Data For Alexnet')
    parser.add_argument("--path", dest="path", help="train and test data dir", default="imgs")
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--output_dir", dest='outd', help=
    "output of yolo results and will be input of alexnet",
                        default="alexnet_data", type=str)
    return parser.parse_args()

#加载yolo模型
args = arg_parse()
batch_size = int(args.bs)#batch size
confidence = float(args.confidence)#过虑初检测的目标的置信度
nms_thesh = float(args.nms_thresh)#NMS阈值
start = 0#python 语法，保证变量可用，无意义
CUDA = torch.cuda.is_available()#GPU是否可用

num_classes = 80#yolo的目标类
classes = load_classes("coco.names")#yolo目标类的名称

#Set up the neural network
print("Loading network.....")#读取模型
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

#如果CUDA可用的话，使用GPU ，否则使用CPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
model.eval()

import os
path =args.path
count = 0
for fn in os.listdir(path): #fn 表示的是文件名
    if os.path.isdir(os.path.join(path,fn)):#检测是否是文件夹
        data = []  # 特征
        label = []  # 标签，车辆类别
        if fn=='train' or fn=='test' :#如果命名为train或test
            for fnn in os.listdir(os.path.join(path,fn)):#遍历文件
                print('当前文件夹:{}/{}'.format(fn,fnn))
                if os.path.isdir(os.path.join(path,fn,fnn)):#因为按车辆类别存储的车辆图片
                    images=os.path.join(path,fn,fnn)#图片路径
                    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]#图片名称列表
                    loaded_ims = [cv2.imread(x) for x in imlist]#使用opencv读取图片
                    im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
                    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
                    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                    leftover = 0
                    if (len(im_dim_list) % batch_size):
                        leftover = 1

                    if batch_size != 1:
                        num_batches = len(imlist) // batch_size + leftover
                        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                                                len(im_batches))])) for i in range(num_batches)]

                    write = 0
                    if CUDA:
                        im_dim_list = im_dim_list.cuda()
                    for i, batch in enumerate(im_batches):
                            if CUDA:
                                batch = batch.cuda()
                            with torch.no_grad():
                                prediction = model(Variable(batch), CUDA)
                            prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)
                            if type(prediction) == int:
                                for im_num, image in enumerate(
                                        imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
                                    im_id = i * batch_size + im_num
                                continue
                            prediction[:, 0] += i * batch_size
                            if not write:
                                output = prediction
                                write = 1
                            else:
                                output = torch.cat((output, prediction))
                            for im_num, image in enumerate(
                                imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
                                im_id = i * batch_size + im_num
                                objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
                            if CUDA:
                                torch.cuda.synchronize()
                    try:
                            output
                    except NameError:
                            print("No detections were made")
                            exit()
                    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
                    scaling_factor = torch.min(416 / im_dim_list, 1)[0].view(-1, 1)
                    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
                    output[:, 1:5] /= scaling_factor
                    for i in range(output.shape[0]):
                        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
                        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])


                    def check_car(label):
                        """
                        :param label: 输入Yolo检测的英文标志
                        :return: 返回是否是车辆的判断
                        """
                        car_labels = ['car', 'truck']
                        if label in car_labels:
                            return True
                        else:
                            return False


                    def get_roi(x, results):
                        """
                        返回感兴趣区域
                        :param x:
                        :param results:
                        :param imlist:
                        :return:
                        """
                        c1 = tuple(x[1:3].int())  # c1为方框左上角坐标x1,y1
                        c2 = tuple(x[3:5].int())  # c2为方框右下角坐标x2,y2
                        img = results[int(x[0])]  # 在results中找到x方框所对应的图片，x[0]为方框所在图片在所有测试图片中的序号
                        cls = int(x[-1])
                        label = "{0}".format(classes[cls])  # label为这个框所含目标类别名字的字符串
                        img_roi_shrink=''#如果图片上没有检测数感兴趣区域，则返回为''后续回去去除
                        if (check_car(label)):
                            # 提取感兴趣区域
                            img_roi_np = img[c1[1]:c2[1], c1[0]:c2[0], :]
                            # 转化为opencv中的Mat格式
                            img_roi_cv=cv2.merge([img_roi_np[:, :, 0], img_roi_np[:, :, 1], img_roi_np[:, :, 2]])
                            # 缩放图像
                            size=(224,224)#alexnet输入为 224X224X3 实际上会经过预处理变为227X227X3
                            img_roi_shrink = cv2.resize(img_roi_cv, size, interpolation=cv2.INTER_AREA)
                            #cv2.imshow('img_roi', img_roi_shrink)
                            #cv2.waitKey()
                        return img_roi_shrink


                    imgs_roi=list(map(lambda x: get_roi(x, loaded_ims), output))#ROI列表
                    imgs_roi_=[i for i in imgs_roi if i is not '']#没有检测出的空元素
                    data.extend(imgs_roi_)#图片数据保存到data
                    label.extend([fnn for i in imgs_roi_])#文件夹名称即图片类比
        else:
            raise ('No train or test dir!')
        np.save('{}/{}_data.npy'.format(args.outd,fn),np.array(data))
        np.save('{}/{}_label.npy'.format(args.outd,fn), np.array(label))
        #print([d.shape for d in data])
        print ('Make {} data done!'.format(fn))

