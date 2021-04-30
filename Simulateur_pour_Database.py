#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time
import pybullet as p
import cv2

import tensorflow as tf
import numpy as np
import math

from qibullet import SimulationManager
from qibullet import PepperVirtual
from qibullet import NaoVirtual
from qibullet import RomeoVirtual

def travel(X, Y, positionX, positionY, robot):
    vectX = X - positionX
    vectY = Y - positionY
    norm = math.sqrt(math.pow(vectX,2)+math.pow(vectY,2))
    thet = math.acos(vectX/norm)
    if(vectX>0):
        thet=-thet
    robot.moveTo(positionX, positionY, theta=thet, frame=1, _async=False, speed=5)
    robot.moveTo(X, Y, theta=thet, frame=1, _async=False, speed=5)
    robot.moveTo(X, Y, theta=0, frame=1, _async=False, speed=5)
    return X,Y

def behave(decisions) :

    if (decisions == '0'):            #waiting
        robot.setAngles('LShoulderPitch', 1.3, 0.8)
        robot.setAngles('RShoulderPitch', 1.3, 0.8)
        time.sleep(0.05)
   
    if (decisions == '1'):            #balle
        print("Detection de la balle")
        robot.setAngles('HeadPitch', -0.2, 0.8)
        robot.setAngles('LShoulderPitch', 1.3, 0.8)
        robot.setAngles('RShoulderPitch', -0.5, 0.8)
        robot.setAngles('RWristYaw', 1.5, 0.8)
        robot.setAngles('RHand', 1, 0.8)
        time.sleep(1.0)
   
    if (decisions == '3'):            #teddy
        print("Detection du teddy")
        robot.setAngles('HeadPitch', -0.2, 1)
        robot.setAngles('LShoulderPitch', -1.5, 1)
        robot.setAngles('RShoulderPitch', -1.5, 1)
        robot.setAngles('RHand', 1, 1)
        robot.setAngles('LHand', 1, 1)
        time.sleep(1.0)             #position de peur/surprise


############################################  MAIN  #################################################

if __name__ == "__main__":
    simulation_manager = SimulationManager()

    client = simulation_manager.launchSimulation(gui=True)

    robot = simulation_manager.spawnPepper(client, spawn_ground_plane=True)

    time.sleep(1.0)
    joint_parameters = list()

    for name, joint in robot.joint_dict.items():
        if "Finger" not in name and "Thumb" not in name:
            joint_parameters.append((
                p.addUserDebugParameter(
                    name,
                    joint.getLowerLimit(),
                    joint.getUpperLimit(),
                    robot.getAnglesPosition(name)),
                name))

    p.connect(p.DIRECT)
    sphere_visual = p.createVisualShape(p.GEOM_SPHERE,radius=0.1,rgbaColor=[1,0,0,1])
    sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
    sphere_body = p.createMultiBody( baseMass=10.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition = [2,1, 0.725])

    duck = p.loadURDF("./duck/duck_vhacd.urdf", basePosition=[-5,-5,0], globalScaling=10)

    handle2 = robot.subscribeCamera(PepperVirtual.ID_CAMERA_TOP)
    ball_detector = cv2.CascadeClassifier('./Haar/data/cascade.xml')
    duck_detector = cv2.CascadeClassifier('./duck/data/cascade.xml')

    #Initialisation
    for joint_parameter in joint_parameters:
        robot.setAngles( joint_parameter[1],p.readUserDebugParameter(joint_parameter[0]), 1.0)

    try:
        while True:
            img2 = robot.getCameraFrame(handle2)
            ball = ball_detector.detectMultiScale(img2, scaleFactor=1.2, minNeighbors=250)
            duck_d = duck_detector.detectMultiScale(img2, scaleFactor=1.05, minNeighbors=50)

            #Detection
            nothing = False #Detection variable

            try :
                x = duck_d[0][0]
                y = duck_d[0][1]
                w = duck_d[0][2]
                h = duck_d[0][3]

                centre_x = int((x+w/2))
                centre_y = int((y+h/2))
                cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.circle(img2,(centre_x,centre_y),1,(0,0,255),-1)

            except : 

                print("no duck detected")

            try :
                x = ball[0][0]
                y = ball[0][1]
                w = ball[0][2]
                h = ball[0][3]

                detected_ball = '1'

                centre_x = int((x+w/2))
                centre_y = int((y+h/2))
                cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.circle(img2,(centre_x,centre_y),1,(0,0,255),-1)

            except :
                nothing = True
                detected_ball = '0'
                print("nothing detected")

            cv2.imshow("top camera", img2)
            cv2.waitKey(1)

            #Behave
            if nothing != True :
                behave(detected_ball)
                if centre_x < 50 :
                    robot.move(1,0,1)
                if centre_x > (img2.shape[1]-50) :
                    robot.move(1,0,-1)
                else :
                    robot.move(1,0,0)
                time.sleep(0.05)
                    
            if nothing == True :
                robot.stopMove()
                behave(detected_ball)
                time.sleep(0.05)


    except KeyboardInterrupt:
        simulation_manager.stopSimulation(client)

############################################  END  #################################################


