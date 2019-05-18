#coding=utf-8
from alexnet import *
import argparse
from torch.utils.data import Dataset, DataLoader,TensorDataset
def arg_parse():
    """
    Parse arguements to the alexnet module

    """
    parser = argparse.ArgumentParser(description='Train and Test Alexnet using YOLO v3 Detection Data ')
    parser.add_argument("--path", dest="path", help="train and test data dir", default="alexnet_data")
    parser.add_argument("--model_path", dest="model_path", help="model_path", default='alexnet.pkl')
    parser.add_argument("--mapping_dic_path", dest="mdp", help="label mapping dict output path", default="mapping.dic")
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--epochs", dest="epochs", help="epochs", default=100)
    return parser.parse_args()


def load_alexnet_data(args):
    """
    :param path: train and test data dir
    :return: X_train,y_train,X_test,y_test 分别为训练集图片，训练集标签，测试集图片，测试集标签 numpy格式
    """
    #numpy 加载数据
    X_train=np.load('{}/train_data.npy'.format(args.path))
    X_test = np.load('{}/test_data.npy'.format(args.path))
    y_train=np.load('{}/train_label.npy'.format(args.path))
    y_test = np.load('{}/test_label.npy'.format(args.path))
    print('训练集图片数量：',X_train.shape[0])
    print('测试集图片数量：', X_test.shape[0])
    #制作车型文字与标签的映射关系
    y=np.concatenate((y_train,y_test),axis=0)#垂直组合
    names=list(set(list(y)))#去重后车型名称
    values=[i for i in range(len(names))]#对应label数值
    mapping_dic=dict(zip(names,values))#映射关系
    print(mapping_dic)
    #保存映射关系到json中
    import json,codecs
    with codecs.open(args.mdp,'w', 'utf-8') as outf:
        json.dump(mapping_dic, outf, ensure_ascii=False)
        outf.write('\n')
    #转化车型文字为数字标签
    y_train=np.array([mapping_dic[i] for i in list(y_train)])
    y_test = np.array([mapping_dic[i] for i in list(y_test)])
    #将numpy保存的图像格式转化为torch的格式(,224,224,3)->(,3,224,224)即H,W,C->C,H,W
    #print(X_train.shape)
    X_train=np.array([i.transpose((2,0,1)) for i in list(X_train)]).astype(np.float32)/255
    X_test = np.array([i.transpose((2,0,1)) for i in list(X_test)]).astype(np.float32)/255
    return X_train,y_train,X_test,y_test

if __name__ == '__main__':
    #输入参数
    args = arg_parse()
    X_train, y_train, X_test, y_test=load_alexnet_data(args)
    y_train=y_train.astype(np.intp)
    y_test=y_test.astype(np.intp)
    #将numpy转化为torch tensor
    X_train=torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train=torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)
    X_train_in = Variable(X_train)
    X_test_in = Variable(X_test)
    y_train_in=Variable(y_train)
    y_test_in = Variable(y_test)
    train_loader = DataLoader(dataset=TensorDataset(X_train_in,y_train_in), batch_size=args.bs, shuffle=True)#训练数据集生成
    #print(train_loader)
    test_loader = DataLoader(dataset=TensorDataset(X_test_in,y_test_in), batch_size=args.bs)#测试数据集生成
    #模型结构
    model_type = 'pre'
    n_output = 196#总共有196类图
    alexnet = BuildAlexNet(model_type, n_output)
    print(alexnet)
    #训练模型
    optimizer = torch.optim.Adam(alexnet.parameters())#Adam训练
    loss_func = torch.nn.CrossEntropyLoss()#交叉熵损失函数
    for epoch in range(np.intp(args.epochs)):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = alexnet(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
            X_train_in)), train_acc / (len(X_train_in))))
        # evaluation--------------------------------
        alexnet.eval()
        eval_loss = 0.
        eval_acc = 0.
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
            out = alexnet(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            X_test_in)), eval_acc / (len(X_test_in))))
    # 保存模型
    torch.save(alexnet.cpu(), args.model_path)
    alexnet = torch.load(args.model_path)  # 加载alexnet模型
    y_pred=torch.max(alexnet(X_test_in), 1)[1].data.numpy()
    from sklearn.metrics import accuracy_score
    print('acc:',accuracy_score(y_test, y_pred))
