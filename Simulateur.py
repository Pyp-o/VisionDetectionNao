#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time
import pybullet as p
import cv2
#import tensorflow as tf
import numpy as np
import math, random
from qibullet import SimulationManager
from qibullet import PepperVirtual
from qibullet import NaoVirtual
from qibullet import RomeoVirtual
from qibullet import base_controller

def travelTo(X, Y, positionX, positionY, robot):
    # on calcule les coordonnées du vecteur pour ensuite calculer l'angle du robot
    vectX = X - positionX
    vectY = Y - positionY

    # calcul de l'angle
    norm = math.sqrt(math.pow(vectX,2)+math.pow(vectY,2))
    thet = math.acos(vectX/norm)

    #en fonction de l'orientation du robot, on aura un angle positif, ou negatif
    if(vectX>0 and vectY<0) or (vectX<0 and vectY<0):
        thet=-thet

    # rotation du robot avant deplacement
    robot.moveTo(positionX, positionY, theta=thet, frame=1, _async=False, speed=5)

    # deplacement
    robot.moveTo(X, Y, theta=thet, frame=1, _async=False, speed=5)
    return X,Y,thet


def decision(ball, duck):
    if duck:
        return 2
    elif ball:
        return 1
    else :
        return 0


def behave(decisions, positionX, positionY, robot, angle) :
    if (decisions == '0'):
        print("Deplacement aléatoire")

        # generation d'une position aléatoire
        Xrand=(random.randint(-3, 3))
        Yrand=(random.randint(-3, 3))

        #calcul de la nouvelle position
        X = Xrand + positionX
        Y = Yrand + positionY

        # on utilise cette fonction pour calculer l'angle que doit adopter le robot pour faire face à son déplacement.
        X, Y, angle = travelTo(X, Y, positionX, positionY, robot)
        return X,Y,angle

    elif (decisions == '1'):
        print("Suivi de balle")

        # position ou le robot veut attraper la balle
        robot.setAngles('HeadPitch', -0.2, 0.8)
        robot.setAngles('LShoulderPitch', 1.3, 0.8)
        robot.setAngles('RShoulderPitch', -0.5, 0.8)
        robot.setAngles('RWristYaw', 1.5, 0.8)
        robot.setAngles('RHand', 1, 0.8)

        # on stocke l'ancien angle pour faciliter les calculs par la suite
        oldAngle = angle

        # on garde un angle compris enter 0 et 2PI
        if (angle >= (math.pi * 2)):
            angle = angle - (math.pi * 2)
        elif (angle <= -(math.pi * 2)):
            angle = angle + (math.pi * 2)

        # rotation du robot en position de fuite
        robot.moveTo(positionX, positionY, theta=angle, frame=1, _async=False, speed=5)

        # calcul du déplacement nécessaire en X et Y
        X = math.cos(oldAngle)
        Y = math.cos((math.pi / 2) - oldAngle)

        # calcul de la nouvelle position
        positionX = positionX + X
        positionY = positionY + Y

        # deplacement du robot à l'endroit voulu en faisant face à la ou il va (dos au canard)
        robot.moveTo(positionX, positionY, theta=angle, frame=1, _async=False, speed=5)
        return positionX, positionY, angle

        return positionX, positionY, angle

    elif (decisions == '2'):
        print("Fuite du canard")

        #position de peur
        robot.setAngles('HeadPitch', -0.2, 1)
        robot.setAngles('LShoulderPitch', -1.5, 1)
        robot.setAngles('RShoulderPitch', -1.5, 1)
        robot.setAngles('RHand', 1, 1)
        robot.setAngles('LHand', 1, 1)

        # on stocke l'ancien angle pour faciliter les calculs par la suite
        oldAngle = angle

        #limitation de la valeur de l'angle
        if(angle>=0 and angle<=math.pi):
            angle = angle - math.pi
        else :
            angle = angle + math.pi

        # on garde un angle compris enter 0 et 2PI
        if (angle >= (math.pi*2)):
            angle = angle - (math.pi * 2)
        elif (angle <= -(math.pi*2)) :
            angle = angle + (math.pi * 2)

        # rotation du robot en position de fuite
        robot.moveTo(positionX, positionY, theta=angle, frame=1, _async=False, speed=5)

        # calcul du déplacement nécessaire en X et Y
        X = -2*math.cos(oldAngle)
        Y = -2*math.cos((math.pi/2)-oldAngle)

        # calcul de la nouvelle position
        positionX = positionX + X
        positionY = positionY + Y

        # deplacement du robot à l'endroit voulu en faisant face à la ou il va (dos au canard)
        robot.moveTo(positionX, positionY, theta=angle, frame=1, _async=False, speed=5)
        return positionX, positionY, angle

    else :
        return positionX, positionY, angle


if __name__ == "__main__":
    simulation_manager = SimulationManager()
    positionX=0
    positionY=0
    angle=0.0

    if (sys.version_info > (3, 0)):
        #rob = input("Which robot should be spawned? (pepper/nao/romeo): ")
        rob = "pepper"
    else:
        #rob = raw_input("Which robot should be spawned? (pepper/nao/romeo): ")
        rob = "pepper"

    client = simulation_manager.launchSimulation(gui=True)

    if rob.lower() == "nao":
        robot = simulation_manager.spawnNao(client, spawn_ground_plane=True)
    elif rob.lower() == "pepper":
        robot = simulation_manager.spawnPepper(client, spawn_ground_plane=True)
    elif rob.lower() == "romeo":
        robot = simulation_manager.spawnRomeo(client, spawn_ground_plane=True)
    else:
        print("You have to specify a robot, pepper, nao or romeo.")
        simulation_manager.stopSimulation(client)
        sys.exit(1)

    time.sleep(0.2)
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
    sphere_visual = p.createVisualShape(p.GEOM_SPHERE,radius=0.1,rgbaColor=[0,0,1,1])
    sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
    sphere_body = p.createMultiBody( baseMass=10.0, baseCollisionShapeIndex=sphere_collision,baseVisualShapeIndex=sphere_visual, basePosition = [2,1, 0.725])
    duck = p.loadURDF("./duck/duck_vhacd.urdf", basePosition=[2, -2, 0], globalScaling=10)
    #p.loadURDF("./urdf/table/table.urdf", basePosition = [2,1,0], globalScaling = 1)
    #p.loadURDF("./urdf/chair/chair.urdf", basePosition = [3,1,0], globalScaling = 1)
    #p.loadURDF("./urdf/chair/chair.urdf", basePosition = [4,1,0], globalScaling = 1)

    # camera
    #handle = robot.subscribeCamera(PepperVirtual.ID_CAMERA_BOTTOM)
    handle2 = robot.subscribeCamera(PepperVirtual.ID_CAMERA_TOP)
    #handle3 = robot.subscribeCamera(PepperVirtual.ID_CAMERA_DEPTH)

    #Laser
    #robot.showLaser(True)
    #robot.subscribeLaser()

    #test
    '''
    robot.goToPosture("Crouch", 0.6)
    time.sleep(3)
    robot.goToPosture("Stand", 0.6)
    time.sleep(3)
    robot.goToPosture("StandZero", 0.6)
    time.sleep(5)
    '''
    #p.loadURDF("./urdf/bullet3-master/data/duck_vhacd.urdf",basePosition=[1, 0, 0.5],globalScaling=10.0)#,physicsClientId=client

    print("---------------Please be patient, work in progress---------------")
    #new_model = tf.keras.models.load_model('./Saved_Model/myModel')
    print("---------------Thanks for waiting, robot is moving now---------------")

    try:
        while True:
            for joint_parameter in joint_parameters:
                robot.setAngles( joint_parameter[1],p.readUserDebugParameter(joint_parameter[0]), 1.0)
            img2 = robot.getCameraFrame(handle2)
            cv2.imshow("top camera", img2)
            cv2.waitKey(1)
            predicti = input("Quel comportement à tester ? (0/1/2/3):")
            #decision(prediction(img2, new_model))
            positionX, positionY, angle = behave(predicti, positionX, positionY, robot, angle)

    except KeyboardInterrupt:
        simulation_manager.stopSimulation(client)
