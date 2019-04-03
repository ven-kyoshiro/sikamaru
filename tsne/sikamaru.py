# -*- coding:utf-8 -*-
import pygame
from pygame.locals import *
import sys

def l2(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**(0.5)

class Sikamaru(object):
    def __init__(self,posi):
        im1 = pygame.image.load(
               "sikamaru1.png").convert_alpha()
        im2 = pygame.image.load(
                "sikamaru2.png").convert_alpha()
        im3 = pygame.image.load(
                "sikamaru3.png").convert_alpha()
        im4 = pygame.image.load(
                "sikamaru4.png").convert_alpha()
        self.origin_im1 = im1
        self.origin_im2 = im2
        self.origin_im3 = im3
        self.origin_im4 = im4
        self._size  = (10.*5,7.*5)
        self.im1 = pygame.transform.smoothscale(im1,self.size_int)
        self.im2 = pygame.transform.smoothscale(im2,self.size_int)
        self.im3 = pygame.transform.smoothscale(im3,self.size_int)
        self.im4 = pygame.transform.smoothscale(im4,self.size_int)
        self.head = 'left'
        self.posi = posi
        self.show_im = 1
        self.rect = self.im1.get_rect()
        self.rect.center = self.posi
        self.smile=False

    @property
    def size_int(self):
        return (int(self._size[0]),int(self._size[1]))

    def get_im(self):
        if self.show_im == 1:
            self.show_im = 2
            if self.smile:
                if self.head == 'left':
                    return self.im3
                else:
                    return pygame.transform.flip(self.im3,True,False)
            else:
                if self.head == 'left':
                    return self.im1
                else:
                    return pygame.transform.flip(self.im1,True,False)
        else:
            self.show_im = 1
            if self.smile:
                if self.head == 'left':
                    return self.im4
                else:
                    return pygame.transform.flip(self.im4,True,False)
            else:
                if self.head == 'left':
                    return self.im2
                else:
                    return pygame.transform.flip(self.im2,True,False)

    def bigup(self):
        self._size = (1.2*self._size[0],1.2*self._size[1])
        self.im1 = pygame.transform.smoothscale(self.origin_im1,
                                                self.size_int)
        self.im2 = pygame.transform.smoothscale(self.origin_im2,
                                                self.size_int)
        self.im3 = pygame.transform.smoothscale(self.origin_im3,
                                                self.size_int)
        self.im4 = pygame.transform.smoothscale(self.origin_im4,
                                                self.size_int)
 
        self.rect = self.im1.get_rect()
        self.rect.center = self.posi
    def move(self,next_posi):
        if next_posi[0] - self.posi[0]<0 and self.head == 'right':
            self.head = 'left'
        elif next_posi[0] - self.posi[0]>0 and self.head == 'left':
            self.head = 'right'
        self.posi = next_posi
        self.rect.center = self.posi


class Food(object):
    def __init__(self):
        im = pygame.image.load(
               "esa.png").convert_alpha()
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
        if l2(sika.posi,self.posi) < sika._size[1]*0.8:
            self.life = 0
            return 1.0
        else:
            self.life-=1
            return 0.0
