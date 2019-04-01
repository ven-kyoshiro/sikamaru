# -*- coding:utf-8 -*-
import pygame
from pygame.locals import *
import sys

class Sikamaru(object):
    def __init__(self,posi):
        im1 = pygame.image.load(
               "../image/sikamaru1.png").convert_alpha()
        im2 = pygame.image.load(
                "../image/sikamaru2.png").convert_alpha()
        self.size  = (100,70)
        self.im1 = pygame.transform.smoothscale(im1,self.size)
        self.im2 = pygame.transform.smoothscale(im2,self.size)
        self.head = 'left'
        self.posi = posi
        self.show_im = 1
        self.rect = self.im1.get_rect()
        self.rect.center = self.posi
    def get_im(self):
        if self.show_im == 1:
            self.show_im = 2
            return self.im1
        else:
            self.show_im = 1
            return self.im2

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
        
def main():
    pygame.init() # 初期化
    (w, h) = (480, 320)
    screen = pygame.display.set_mode((w,h)) # window size
    pygame.display.set_caption("Sikamaru") # window bar

    # initialization
    tx = 0
    ty = 0
    sika = Sikamaru((w/2, h/2))
    sleep_count = 5
    eat_mode = 100
    esa = Food()
    wait = True

    # TODO define RL agent
    '''
    state : 4D (sikaposi, esaposi)
    action : 2D (-20,+20)^2
    SAC
    simple_net : 30,30
    
    '''


    while(True):
        screen.fill((0,100,0,)) # backgroud color

        # my procedure
        ## env
        a = rule_act((tx,ty),sika.posi)
        nx = sika.posi[0] + a[0]
        nx = max(min(nx,w),0)
        ny = sika.posi[1] + a[1]
        ny = max(min(ny,h),0)
        
        sika.move((nx,ny))
        screen.blit(sika.get_im(), sika.rect)

        if esa.life: # RL
            # TOOD:record as epi
            
            screen.blit(esa.im, esa.rect)
            rew = esa.life_step(sika)
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

            if event.type == QUIT: # 終了処理
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    main()
