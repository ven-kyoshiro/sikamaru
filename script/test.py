# -*- coding:utf-8 -*-
import pygame
from pygame.locals import *
import sys

# for RL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import machina as mc
from machina.pols import CategoricalPol
from machina.algos import sac
from machina.vfuncs import DeterministicSAVfunc
from machina.envs import GymEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import set_device, measure
from simple_net import PolNet, QNet, VNet


class Sikamaru(object):
    def __init__(self,posi):
        im1 = pygame.image.load(
               "../image/sikamaru1.png").convert_alpha()
        im2 = pygame.image.load(
                "../image/sikamaru2.png").convert_alpha()
        self._size  = (10.*5,7.*5)
        self.im1 = pygame.transform.smoothscale(im1,self.size_int)
        self.im2 = pygame.transform.smoothscale(im2,self.size_int)
        self.head = 'left'
        self.posi = posi
        self.show_im = 1
        self.rect = self.im1.get_rect()
        self.rect.center = self.posi
    @property
    def size_int(self):
        return (int(self._size[0]),int(self._size[1]))
    def get_im(self):
        if self.show_im == 1:
            self.show_im = 2
            return self.im1
        else:
            self.show_im = 1
            return self.im2
    def bigup(self):
        self._size = (1.001*self._size[0],1.001*self._size[1])
        self.im1 = pygame.transform.smoothscale(self.im1,self.size_int)
        self.im2 = pygame.transform.smoothscale(self.im2,self.size_int)
        self.rect = self.im1.get_rect()
        self.rect.center = self.posi
    def move(self,next_posi):
        if next_posi[0] - self.posi[0]<0 and self.head == 'right':
            self.im1 = pygame.transform.flip(self.im1,True,False)
            self.im2 = pygame.transform.flip(self.im2,True,False)
            self.head = 'left'
        elif next_posi[0] - self.posi[0]>0 and self.head == 'left':
            self.im1 = pygame.transform.flip(self.im1,True,False)
            self.im2 = pygame.transform.flip(self.im2,True,False)
            self.head = 'right'
        self.posi = next_posi
        self.rect.center = self.posi

def l2(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**(0.5)

class Food(object):
    def __init__(self):
        im = pygame.image.load(
               "../image/esa.png").convert_alpha()
        self.size  = (20,20)
        self.im = pygame.transform.smoothscale(im, self.size)
        self.rect = self.im.get_rect()
        self.life = 0

    def set(self,posi):
        self.posi = posi
        self.rect.center = self.posi
        self.life = 100

    def life_step(self,sika):
        self.life-=1
        if l2(sika.posi,self.posi) < 50:
            self.life = 0
            return 1.0
        else:
            self.life-=1
            return 0.0

def rule_act(esaposi,sikaposi):
    ax = 0.1*(esaposi[0] - sikaposi[0]) 
    ay = 0.1*(esaposi[1] - sikaposi[1]) 
    return (ax,ay)

def make_obs(esaposi,sikaposi,w,h):
    return np.array([sikaposi[0]/w,sikaposi[1]/h,esaposi[0]/w,esaposi[1]/h])
        
def main():
    pygame.init() # 初期化
    (w, h) = (480, 320)
    screen = pygame.display.set_mode((w,h),FULLSCREEN) # window size
    pygame.display.set_caption("Sikamaru") # window bar

    # initialization
    tx = 0
    ty = 0
    sika = Sikamaru((w/2, h/2))
    sleep_count = 5
    eat_mode = 100
    esa = Food()
    wait = True
    seed=42

    # TODO define RL agent
    '''
    state : 4D (sikaposi, esaposi)
    action : 2D (-20,+20)^2
    SAC
    simple_net : 30,30
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)   

    low = np.zeros(4)
    high = w*np.ones(4)
    ob_space = gym.spaces.Box(low=low, high=high)
    ac_space = gym.spaces.Discrete(4)
    ac_dict = {0:np.array([-20,0]),
               1:np.array([20,0]),
               2:np.array([0,-20]),
               3:np.array([0,20])
               }
    pol_net = PolNet(ob_space, ac_space)
    pol = CategoricalPol(ob_space, ac_space, pol_net)
    qf_net1 = QNet(ob_space, ac_space)
    qf1 = DeterministicSAVfunc(ob_space, ac_space, qf_net1)
    targ_qf_net1 = QNet(ob_space, ac_space)
    targ_qf_net1.load_state_dict(qf_net1.state_dict())
    targ_qf1 = DeterministicSAVfunc(ob_space, ac_space, targ_qf_net1)
    qf_net2 = QNet(ob_space, ac_space)
    qf2 = DeterministicSAVfunc(ob_space, ac_space, qf_net2)
    targ_qf_net2 = QNet(ob_space, ac_space)
    targ_qf_net2.load_state_dict(qf_net2.state_dict())
    targ_qf2 = DeterministicSAVfunc(ob_space, ac_space, targ_qf_net2)
    qfs = [qf1, qf2]
    targ_qfs = [targ_qf1, targ_qf2]
    log_alpha = nn.Parameter(torch.ones(()))

    optim_pol = torch.optim.Adam(pol_net.parameters(), 1e-4)
    optim_qf1 = torch.optim.Adam(qf_net1.parameters(), 3e-4)
    optim_qf2 = torch.optim.Adam(qf_net2.parameters(), 3e-4)
    optim_qfs = [optim_qf1, optim_qf2]
    optim_alpha = torch.optim.Adam([log_alpha], 1e-4)
    
    # off_traj = Traj()
    

    while(True):
        screen.fill((0,100,0,)) # backgroud color

        # my procedure
        ## env
        obs = make_obs((tx,ty),sika.posi,w,h)
        ac_real, ac, a_i = pol.deterministic_ac_real(torch.tensor(obs, dtype=torch.float))
        # ac_real = ac_real.reshape(pol.ac_space.shape)
        a = rule_act((tx,ty),sika.posi)
        # a = ac_dict[int(ac_real)]

        nx = sika.posi[0] + a[0]
        nx = max(min(nx,w),0)
        ny = sika.posi[1] + a[1]
        ny = max(min(ny,h),0)
        
        sika.move((nx,ny))
        screen.blit(sika.get_im(), sika.rect)

        if esa.life: # RL
            # TOOD:record as epi
            
            screen.blit(esa.im, esa.rect)
            # scr
            rew = esa.life_step(sika)
            if rew>0:
                sika.bigup()
            if esa.life == 0:
                pass
                #TODO add one epi and learn

                wait = False

        if wait:
            pygame.time.wait(500)
        wait = True
        pygame.display.update() # 画面更新


        ## event
        for event in pygame.event.get():
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                tx, ty = event.pos
                esa.set((tx,ty))
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    sys.exit()

            if event.type == QUIT: # 終了処理
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    main()
