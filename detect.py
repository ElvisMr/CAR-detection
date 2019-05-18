#coding=utf-8
from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from alexnet import *
import json
import codecs
def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--slice_dir", dest = 'slicedir', help =
                        "Store slice images",
                        default = "slice", type = str)
    parser.add_argument("--is_fine_classify", dest = 'ifc', help =
                        "wheater to use alexnet to classify cars",
                        default = True)
    parser.add_argument("--model_path", dest="model_path", help="model_path", default='alexnet.pkl')
    parser.add_argument("--mapping_dic_path", dest="mdp", help="label mapping dict output path", default="mapping.dic")
    parser.add_argument("--img_path", dest="imgp", help="singel img to dectet", default=None)
    return parser.parse_args()
    
args = arg_parse()
images = args.images#图片路径
batch_size = int(args.bs)#batch size
confidence = float(args.confidence)#多虑初检测的目标的置信度
nms_thesh = float(args.nms_thresh)#NMS阈值
ifc=args.ifc#是否使用Alexnet分类识别出的car
#alexnet加载
alexnet = torch.load(args.model_path)#加载alexnet模型
#映射关系
with codecs.open(args.mdp, 'r', 'utf-8') as outf:
    mdp = json.loads(outf.read(), strict=False)


start = 0#python 语法，保证变量可用，无意义
CUDA = torch.cuda.is_available()#GPU是否可用


num_classes = 80#yolo的目标类
classes = load_classes("coco.names")#yolo目标类的名称

#Set up the neural network，读取模型
print("Loading network.....")
model = Darknet(args.cfgfile)#配置
model.load_weights(args.weightsfile)#权重
print("Network successfully loaded")
model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 #保证输入图片能够分割成32*32的狂
assert inp_dim > 32

#如果CUDA可用的话，使用GPU ，否则使用CPU
if CUDA:
    model.cuda()

#Set the model in evaluation mode，Pytorch语法
model.eval()

#读取文件夹里面的数据
read_dir = time.time()

#检测设置
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]#图片名称列表
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()
if args.imgp is not None:
    imlist=[args.imgp]
#det保存的是检测后的图像
if not os.path.exists(args.det):
    os.makedirs(args.det)
#slicedir保存的是分割后的小图
def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)
if os.path.exists(args.slicedir):
    del_file(args.slicedir)
if not os.path.exists(args.slicedir):
    os.makedirs(args.slicedir)


load_batch = time.time()
imlist=[i for i in imlist if not os.path.isdir(i) ]#从列表中删除文件夹
loaded_ims = [cv2.imread(x) for x in imlist]#使用opencv读取图片
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)


leftover = 0
if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                        len(im_batches))]))  for i in range(num_batches)]  

write = 0


if CUDA:
    im_dim_list = im_dim_list.cuda()
    
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
#load the image 
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        prediction = model(Variable(batch), CUDA)

    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)

    end = time.time()

    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist 

    if not write:                      #If we have't initialised output
        output = prediction  
        write = 1
    else:
        output = torch.cat((output,prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()       
try:
    output
except NameError:
    print ("No detections were made")
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)


output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2



output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    
output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))
#在图中标记检测的车
draw = time.time()
def check_car(label):
    """
    :param label: 输入Yolo检测的英文标志
    :return: 返回是否是车辆的判断
    """
    car_labels=['car','truck']
    if label in car_labels:
        return True
    else:
        return False

def write(x, results):
    """
    标记提取的车辆区域
    :param x:
    :param results:
    :return:
    """
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    if (check_car(label)):
        if ifc:
            # 提取感兴趣区域
            img_roi_np = img[c1[1]:c2[1], c1[0]:c2[0], :]
            # 转化为opencv中的Mat格式
            img_roi_cv = cv2.merge([img_roi_np[:, :, 0], img_roi_np[:, :, 1], img_roi_np[:, :, 2]])
            # 缩放图像
            size = (224, 224)  # alexnet输入为 224X224X3 实际上会经过预处理变为227X227X3
            img_roi_shrink = cv2.resize(img_roi_cv, size, interpolation=cv2.INTER_AREA).transpose((2,0,1))[np.newaxis,]#由于是一个样本，需要增加一个维度
            # 将numpy转化为torch tensor
            X = torch.from_numpy(img_roi_shrink.astype(np.float32)/255)
            X_in = Variable(X)
            y = torch.max(alexnet(X_in), 1)[1].data.numpy()#具体car name
            print(alexnet(X_in), 1)
            label=list(mdp.keys())[list(mdp.values()).index(y)]
            #label = "car"
        else:
            label = "car"
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img
def slice(x,results,imlist):
    """
    保存感兴趣区域
    :param x:
    :param results:
    :param imlist:
    :return:
    """
    c1 = tuple(x[1:3].int())# c1为方框左上角坐标x1,y1
    c2 = tuple(x[3:5].int())# c2为方框右下角坐标x2,y2
    img = results[int(x[0])]  # 在results中找到x方框所对应的图片，x[0]为方框所在图片在所有测试图片中的序号
    img_name=imlist[int(x[0])].split("/")[-1]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])  # label为这个框所含目标类别名字的字符串
    if (check_car(label)):
        #提取感兴趣区域
        img_roi = img[c1[1]:c2[1],c1[0]:c2[0],:]
        #print(img.shape)
        #print(img_roi.shape)
        file_path = "{}\\slice_{}".format(args.slicedir ,img_name)
        cv2.imwrite(file_path, img_roi)

    return img
list(map(lambda x: slice(x, loaded_ims,imlist), output))#保存感兴趣区域
list(map(lambda x: write(x, loaded_ims), output))#标记提取的车辆区域
det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("\\")[-1]))#检测结果保存地址
print(det_names)
list(map(cv2.imwrite, det_names, loaded_ims))#保存


end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()
