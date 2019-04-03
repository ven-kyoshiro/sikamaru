import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle
from sklearn.manifold import TSNE

import numpy as np
import torch
import torch.nn as nn

import socket
import time
import os

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d

def get_feature(img,sess,sess_args):
    image_v = np.expand_dims((img.astype('float') / 255.0) - 0.5, 0)

    hand_scoremap_v, image_crop_v, scale_v, center_v,\
    keypoints_scoremap_v, keypoint_coord3d_v = sess.run(sess_args,
                                                        feed_dict={image_tf: image_v})
    coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    center_v = np.array([0.,0.])
    scale_v = np.array([1.])
    coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)
    return coord_hw#/100#coord_hw_crop# - center_v

def get_reduced_posi_opt(X, X_reduced,x_new,perp):
    x_new = torch.tensor(x_new, requires_grad=True,dtype=torch.double)
    y_new = torch.tensor([0.0,0.0], requires_grad=True,dtype=torch.double)
    X = torch.tensor(X, requires_grad=True,dtype=torch.double)
    Y = torch.tensor(X_reduced, requires_grad=True,dtype=torch.double)
    sigma = torch.tensor(1.0, requires_grad=True,dtype=torch.double)

    # find sigma
    best_sigma = {'score':10000,'param':9999}
    endure = 3
    endure_count = 0
    for i in range(1000):
        twosigma = torch.tensor(2., requires_grad=True,dtype=torch.double)*torch.pow(sigma,2)
        p = torch.exp(-torch.pow(X - x_new,2).sum(dim=1)/twosigma)
        P = (p/p.sum())#.detach()
        lr_perp = 0.00001
        H = (-P*torch.log2(P)).sum()
        loss_p = torch.norm(torch.tensor(perp, requires_grad=True,dtype=torch.double) - torch.pow(2,H))
        loss_p.backward()
        sigma.data.sub_(lr_perp * sigma.grad.data)
        sigma.grad.data.zero_()

        score = loss_p.detach().numpy()
        sig = sigma.detach().numpy()
        if score < best_sigma['score']:
            best_sigma['score'] = score
            best_sigma['param'] = sig
            endure_count=0
            #print('finding perp:loss== ',score)
        else:
            endure_count+=1
            if endure_count==endure:
                break
    sigma = torch.tensor(best_sigma['param'], requires_grad=True,dtype=torch.double)
    twosigma = torch.tensor(2., requires_grad=True,dtype=torch.double)*torch.pow(sigma,2)
    p = torch.exp(-torch.pow(X - x_new,2).sum(dim=1)/twosigma)
    P = (p/p.sum()).detach()

    # find y_new
    lr = 0.1
    endure_count=0
    best_y_new = {'score':10000,'param':9999}
    for i in range(1000):
        q = torch.ones(1,dtype=torch.double, requires_grad=True)/(torch.ones(1,dtype=torch.double, requires_grad=True) +torch.pow(Y - y_new,2).sum(dim=1))
        Q = q/q.sum()
        loss = (P*torch.log(P/Q)).sum()
        loss.backward()
        y_new.data.sub_(lr * y_new.grad.data)
        y_new.grad.data.zero_()

        score = loss.detach().numpy()
        y_n = y_new.detach().numpy()
        if score < best_y_new['score']:
            best_y_new['score'] = score
            best_y_new['param'] = y_n
            endure_count=0
            #print('finding y_new:loss== ',score)
        else:
            endure_count+=1
            if endure_count==endure:
                break
    return best_y_new['param']

def get_reduced_posi_1nn(X, X_reduced,new_x):
    min_dist = 1000
    min_id = 1000000000
    # get new_position
    for j, d in enumerate(X):
        dist = np.linalg.norm(d - new_x)
        if min_dist > dist:
            min_id = j
            min_dist = dist
    return X_reduced[min_id]

### novel code : add new point
def get_reduced_posi_1nn(X, X_reduced,new_x):
    min_dist = 1000
    min_id = 1000000000
    # get new_position
    for j, d in enumerate(X):
        dist = np.linalg.norm(d - new_x)
        if min_dist > dist:
            min_id = j
            min_dist = dist
    return X_reduced[min_id]

def get_high_low(D_l2,sigmas,perp):
    # print('sig,D_l2:',sigmas.shape,D_l2.shape)
    D = torch.exp(-D_l2/(2*torch.pow(sigmas,2).view((sigmas.shape[0],1)))) -\
                                                torch.eye(D_l2.shape[0],dtype=torch.double)
    P = D/D.sum(dim=1).view((D.shape[0],1))
    H = (-P*torch.log2(P+torch.eye(P.shape[0],dtype=torch.double))).sum(dim=1)
    req_perp = torch.tensor(perp, requires_grad=True,dtype=torch.double)
    diff = torch.pow(2,H) - req_perp
    return -torch.sign(diff),diff
    
SIGMAX = 20
def multi_line_search(D_l2,perp):
    l = torch.zeros((D_l2.shape[0],1),dtype=torch.double)
    h = SIGMAX*torch.ones((D_l2.shape[0],1),dtype=torch.double)
    lh = torch.cat((l,h),dim=1)
    A = torch.tensor([[3,1],[1,3]],dtype=torch.double)
    B = torch.tensor([[-1,1],[-1,1]],dtype=torch.double)
    # losses = []
    for i in range(100):
        x,diff = get_high_low(D_l2,lh.mean(dim=1),perp)
        # losses.append(diff.abs().mean().detach().numpy())
        lh = (A@(lh.transpose(0,1)) + B@((lh*x.view((x.shape[0],1))).transpose(0,1)))/4
        lh = lh.transpose(0,1)
        if diff.abs().max() < 1e-2:
            break
    assert lh.max()<20,'you should make SIGMAX larger'
    return lh.mean(dim=1)# ,losses
def line_search(x,P,p,grad):
    c = 1e-4
    rho = 0.8
    alpha = 100.0
    while True:
        if kl_loss(x+alpha*p,Y,P)<= kl_loss(x,Y,P)+c*alpha*grad@p:
            break
        else:
            alpha *= rho
    return alpha*p

def kl_loss(y_new,Y,P):
    Q = get_Q(Y,y_new)
    loss = (P*torch.log(P/Q)).sum()
    return loss

def make_D_l2(x):
    xx = x.expand(x.shape[0],x.shape[0],x.shape[1])
    return torch.pow(xx - xx.transpose(0,1),2).sum(dim=2)

def get_P(D_l2,sigmas):
    D_ = -D_l2/(2*torch.pow(sigmas,2).view((sigmas.shape[0],1)))
    D = torch.exp(D_) - torch.eye(D_l2.shape[0],dtype=torch.double)
    P_ = (D/(D.sum(dim=1).view((D.shape[0],1))))/D.shape[0]
    return (P_ +P_.transpose(0,1))/2+torch.eye(D_l2.shape[0],dtype=torch.double)

def get_Q(Y,y_new):
    Y_new = torch.cat((Y,y_new.view(1,2)))
    D_ = make_D_l2(Y_new)
    Q_ = (1/(1 + D_ ))- torch.eye(D_.shape[0],dtype=torch.double)
    return Q_/Q_.sum() + torch.eye(D_.shape[0],dtype=torch.double)

def find_y_new(P,Y,y_0=[0, 0]):
    y_new = torch.tensor(y_0, requires_grad=True,dtype=torch.double)
    for _ in range(100):
        loss = kl_loss(y_new,Y,P)
        loss.backward()
        grad = y_new.grad
        p = - grad
        stp = line_search(y_new,P,p,grad)
        y_new.data.add_(stp)
        y_new.grad.data.zero_()
        if stp.abs().sum()<1e-3:
            break  
    return y_new# ,log


def get_y_new(X,Y,x_new,perp):
    X_new = torch.cat((X,x_new.view(1,42)))
    D_l2 = make_D_l2(X_new)
    sigmas = multi_line_search(D_l2,perp)
    P = get_P(D_l2,sigmas).detach()
    y_new = torch.tensor([0,0], requires_grad=True,dtype=torch.double)
    y_new = find_y_new(P,Y,y_0=get_reduced_posi_1nn(X.detach().numpy(),Y.detach().numpy(),x_new.detach().numpy()))
    return y_new

#### main
perp = 30
mb = '../mb'
port = 7777

# preprocess
# network input
image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
evaluation = tf.placeholder_with_default(True, shape=())
# build network
net = ColorHandPose3DNetwork()
hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)
# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# initialize network
net.init(sess)

# preprare tsne
with open('mukai_dataset.pickle',mode='rb') as f:
    data_set = pickle.load(f) # 100で割ってあるやつ
X_reduced = TSNE(n_components=2, random_state=0, perplexity=perp).fit_transform(data_set)
X = torch.tensor(data_set, requires_grad=True,dtype=torch.double)
Y = torch.tensor(X_reduced, requires_grad=True,dtype=torch.double)

ini_set = {'area_min':X_reduced.min(axis=0),
          'area_max':X_reduced.max(axis=0)}
with open(mb+'/initial_setting.pickle', mode='wb') as f:
    pickle.dump(ini_set,f)

# server mode
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", port))
while True:
    print("waitnig... port:{}".format(port))  
    s.listen(1)
    cli, addr = s.accept()
    print('connected from ',str(addr))
    data = np.array([])
    for _ in range(113):
        new = cli.recv(8*2048)
        new = np.fromstring(new,dtype=float)
        data= np.hstack((data,new))
    img = data.reshape((240, 320, 3))
    hands_2d = get_feature(img,sess,[hand_scoremap_tf, image_crop_tf, 
                                     scale_tf, center_tf,keypoints_scoremap_tf,
                                     keypoint_coord3d_tf])
    # new_posi2 = get_reduced_posi_1nn(data_set, X_reduced,hands_2d.flatten()/100)
    # new_posi  = get_reduced_posi_opt(data_set, X_reduced,hands_2d.flatten()/100,perp)
    x_new = torch.tensor(hands_2d.flatten()/100, requires_grad=True,dtype=torch.double)
    new_posi  = get_y_new(X,Y,x_new,perp).detach().numpy()
    
    print(new_posi)
    cli.send(new_posi.tostring())
