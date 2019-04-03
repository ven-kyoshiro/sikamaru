# -*- coding:utf-8 -*-
import pygame
import sys
from pygame.locals import *
import numpy as np
import time
from sikamaru import Sikamaru, Food
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--disp",default='std')
args = parser.parse_args()

def get_new_y():
    rad = (int(time.time())%20)/20*2*np.pi
    return [np.sin(rad),np.cos(rad)]
def adj_new_y(xy,mx,mn):
    adj_x = (xy[0]-mn[0])/(mx[0]-mn[0])*260+60
    adj_y = (xy[1]-mn[1])/(mx[1]-mn[1])*260+60
    return [int(adj_x),int(adj_y)]


# server
import os
import pickle
mb = 'message_box'
os.system('scp ***** /initial_setting.pickle .')
if os.path.exists('initial_setting.pickle'):
    with open('initial_setting.pickle',mode='rb') as f:
        ini_set = pickle.load(f)
else:
    print('the initial_setting is not found')

import cv2
import os
import time
import matplotlib.pyplot as plt
def get_image(cap):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[:,160:-160,:]
    frame = cv2.resize(frame , (int(frame.shape[1]/3), int(frame.shape[0]/3)))
    return frame

def get_new_posi(img,cli,port):
    img.astype(float)
    cli.send(img.astype(float).tostring())
    data = cli.recv(2048)
    print(np.fromstring(data,dtype=float))
    return np.fromstring(data,dtype=float)


cap = cv2.VideoCapture(0)
img = get_image(cap)

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import numpy as np
import socket

port = 7777

mode = 'control_sika'
(w, h) = (640, 320)
if args.disp=='full':
    screen = pygame.display.set_mode((w,h) ,FULLSCREEN) # window size
else:
    screen = pygame.display.set_mode((w,h))# ,FULLSCREEN) # window size
pygame.display.set_caption("Sikamaru") # window bar
tx = 10
ty = 10
sika = Sikamaru((480, 160))
esa = Food()


# items
buttons = dict(
        radius=10,
        posi_y=[90+i*48 for i in range(5)],
        posi_x=30,
        col_list=[
        (255,100,100),
        (100,255,100),
        (100,100,255),
        (255,255,255),
        (150,150,150)])


modes = ['draw_0','draw_1','draw_2','eraser','sika']
control_dict = {}
for m in modes[:-2]: 
    control_dict[m]=[]
old_y=[]

act_mat = [[np.cos(np.pi/2),-np.sin(np.pi/2)],
           [np.cos(np.pi/2+np.pi*4/3),-np.sin(np.pi/2+np.pi*4/3)],
           [np.cos(np.pi/2+np.pi*2/3),-np.sin(np.pi/2+np.pi*2/3)]]
act_mat = np.array(act_mat).T

while True:
    screen.fill((0,100,0,)) # backgroud color
    # communication
    if mode == 'control_sika':
        img = get_image(cap)
        cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cli.connect(("127.0.0.1", port))
        old_y.append(adj_new_y(get_new_posi(img,cli,port),
            [ini_set['area_max'][0]+1.2,ini_set['area_max'][1]+1.2],
            [ini_set['area_min'][0]-1.2,ini_set['area_min'][1]-1.2]))
        cli.close()

    # calc rule
    screen.blit(sika.get_im(), sika.rect)
    if mode == 'control_sika':
        flug = np.array([0,0,0])
        for m in modes[:-2]:
            for xy in control_dict[m]:
                if (xy[0]-old_y[-1][0])**2+(xy[1]-old_y[-1][1])**2<\
                        buttons['radius']**2:
                    flug[int(m[-1])]=1.0
                    break

        vxvy = act_mat @ flug
        nx = sika.posi[0] + vxvy[0]*20
        nx = max(min(nx,640),320)
        ny = sika.posi[1] + vxvy[1]*20
        ny = max(min(ny,320),0)
        
        sika.move((nx,ny))

    # draw shapes
    pygame.draw.rect(screen, (100,100,100), Rect(0,0,320,320))
    pygame.draw.rect(screen, (128,128,128), Rect(60,60,260,260))

    mini_b = [[0,-8],
              [6,4],
              [-6,4],
              [0,0],
              [0,0]]

    for c,py,mib,m in zip(buttons['col_list'],buttons['posi_y'],mini_b,modes):
        pygame.draw.circle(screen, c, (buttons['posi_x'],py), buttons['radius'])
        pygame.draw.circle(screen, (50,50,50), 
                (buttons['posi_x']+mib[0],py+mib[1]), 3)
        if 'draw' in m:
            for xy in control_dict[m]:
                pygame.draw.circle(screen, c, 
                        xy, buttons['radius'])
    pygame.draw.circle(screen, (0,0,0), (tx,ty), 1)
    # add plots
    for y in old_y:
        pygame.draw.circle(screen, (220,220,220), y, 1)
    pygame.draw.circle(screen, (0,0,0), old_y[-1], 4)
    pygame.draw.circle(screen, (240,240,0), old_y[-1], 3)

    # esa rule
    if esa.life==0:
        po = np.random.random(2)*320+np.array([320,0])
        esa.set([int(p) for p in po])
        # esa.set([100,100])
    else:
        screen.blit(esa.im, esa.rect)
        if esa.life_step(sika):
            sika.bigup()
            sika.smile=True
        else:
            sika.smile=False

    screen.blit(esa.im, esa.rect)
    if sika._size[1]>1000:
        earth = pygame.image.load(
              "earth.jpg").convert_alpha()
        earth = pygame.transform.smoothscale(earth,(640,320))
        sika = Sikamaru((360, 160))
        sika.smile = True
        for i in range(15):
            sika.bigup()
            screen.blit(earth, earth.get_rect())
            screen.blit(sika.get_im(), sika.rect)
            pygame.time.wait(1000)
            pygame.display.update() # 画面更新
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        sys.exit()

                if event.type == QUIT: # 終了処理
                    pygame.quit()
                    sys.exit()
        sika = Sikamaru((480, 160))
        
    pygame.time.wait(50)
    pygame.display.update() # 画面更新


    ## event
    for event in pygame.event.get():
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            tx, ty = event.pos
            # chage mode and draw shap
            # クリックされたのがボタンだったとき
            # 強制的にそのモードになって終わり
            for m,c,py in zip(modes,buttons['col_list'],buttons['posi_y']):
                if (ty-py)**2+(tx-buttons['posi_x'])**2 < buttons['radius']**2:
                    if m == 'sika':
                        mode='control_sika'
                        buttons['col_list']=[
                            (255,100,100),
                            (100,255,100),
                            (100,100,255),
                            (255,255,255),
                            (150,150,150)]
                    elif m == 'eraser':
                        mode = 'eraser'
                        buttons['col_list']=[
                            (255,100,100),
                            (100,255,100),
                            (100,100,255),
                            (200,200,200),
                            (150,150,150)]
                    else:
                        buttons['col_list']=[
                            (255,100,100),
                            (100,255,100),
                            (100,100,255),
                            (255,255,255),
                            (150,150,150)]
                        mode=m
                        if '0' in mode:
                            buttons['col_list'][0]=(255,180,180)
                        elif '1' in mode:
                            buttons['col_list'][1]=(180,255,180)
                        elif '2' in mode:
                            buttons['col_list'][2]=(180,180,255)

 
            print(mode)
            if 'draw' in mode:
                if all([60+10 < tx,tx<320-10,60+10<ty,ty<320-10]):
                    control_dict[mode].append([tx,ty])
                    
            elif mode == 'eraser':
                for m in modes[:-2]:
                    for i in range(len(control_dict[m])):
                        if (control_dict[m][i][0]-tx)**2+(control_dict[m][i][1]-ty)**2 < buttons['radius']**2:
                            del control_dict[m][i]
                            break

        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                sys.exit()

        if event.type == QUIT: # 終了処理
            pygame.quit()
            sys.exit()
