from __future__ import print_function

import time

from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
from numpy import *
import threading
import math as mt
import socket
import json


def arm_and_takeoff(aTargetAltitude):
    print("Basic pre-arm checks")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)
    print("Arming motors")
    vehicle.mode = VehicleMode("MANUAL")
    vehicle.armed = True
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)
        print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)
    time.sleep(1)


def YAW_SPEED(yaw, speed, relative):
    if relative:
        isRelative = 1
    else:
        isRelative = -1
    msg = vehicle.message_factory.command_long_encode(
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_CMD_NAV_SET_YAW_SPEED,  # command
        0,  # confirmation
        yaw,
        speed,
        isRelative,
        0, 0, 0, 0)
    vehicle.send_mavlink(msg)
    vehicle.flush()


def set_home(lat, lon, alt):
    msg = vehicle.message_factory.command_long_encode(
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_CMD_DO_SET_HOME,  # command
        0,  # confirmation
        0,  # Use current (1=use current location, 0=use specified location)
        0, 0,  # empty
        0,  # Yaw
        lat, lon, alt)
    vehicle.send_mavlink(msg)
    vehicle.flush()


def send_nav_velocity(lat, lon, alt):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
        0b0000111111111000,  # type_mask (only speeds enabled)
        lat, lon, alt,  # x, y, z positions (not used)
        0, 0, 0,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not used)
        0, 0)  # yaw, yaw_rate (not used)
    vehicle.send_mavlink(msg)
    vehicle.flush()


def distance(point, position):
    a = ((point[0] - position[0]) ** 2 + (point[1] - position[1]) ** 2) ** 0.5
    return a


def send_message(clientSockets, vehicles):
    global c
    while 1:
        message = [1,
                   vehicles.airspeed,
                   [vehicles.location.global_frame.lon, vehicles.location.global_frame.lat],
                   vehicles.heading,
                   [float("%.2f" % vehicles.location.local_frame.east),
                    float("%.2f" % vehicles.location.local_frame.north)]]
        message_json = json.dumps(message, ensure_ascii=False)
        clientSockets.send(message_json.encode('UTF-8'))
        time.sleep(0.25)


def send_suanfa(clientSockets):
    global vs
    global r
    global H
    global k
    global c
    while 1:
        message1 = [2,
                    vs,
                    r,
                    H,
                    k,
                    c]
        message_json = json.dumps(message1, ensure_ascii=False)
        clientSockets.send(message_json.encode('UTF-8'))
        time.sleep(1)


def send_require(clientSockets):
    message2 = [3]
    message_json = json.dumps(message2, ensure_ascii=False)
    clientSockets.send(message_json.encode('UTF-8'))


def recv_message(clientSocket):
    global cs
    global vs
    global r
    global H
    global k
    global vehicle
    global suanfa
    while 1:
        try:
            recv_messages = clientSocket.recv(1024)
            recv_messages = json.loads(recv_messages.decode("UTF-8"), strict=False)
        except:
            continue
        if recv_messages[0] == 1:
            cs[0] = recv_messages[1]
            cs[1] = recv_messages[2]
            cs[2] = recv_messages[3]
        elif recv_messages[0] == 2:
            vs = recv_messages[1]
            r = recv_messages[2]
            H = recv_messages[3]
            k = recv_messages[4]
        elif recv_messages[0] == 3:
            set_home(recv_messages[1], recv_messages[2], 0)
        elif recv_messages[0] == 4:
            points[0] = recv_messages[1]
            points[1] = recv_messages[2]
        elif recv_messages[0] == 5:
            suanfa = 1
            vehicle.mode = VehicleMode(recv_messages[1])
        elif recv_messages[0] == 6:
            suanfa = 0
            vehicle.mode = VehicleMode('GUIDED')
            send_nav_velocity(vehicle, recv_messages[1], recv_messages[2])


if __name__ == '__main__':
    connection_string = '127.0.0.1:14570'  # /dev/serial0
    print('Connecting to vehicle on: %s' % connection_string)
    # vehicle = connect(connection_string,wait_ready=True,baud=921600)#baud=921600
    vehicle = connect('127.0.0.1:14570', wait_ready=True)
    arm_and_takeoff(-1)

    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientSocket.bind(('192.168.253.128', 9992))
    clientSocket.connect(('192.168.253.1', 10000))

    c = 0.0
    cs = [0.0, 0.0, 0.0]
    suanfa = 0
    vs = 1.0
    H = 2.0
    r = 6.0
    k = 0.16
    points = [0.0, 0.0]
    thread = threading.Thread(target=send_message, args=(clientSocket, vehicle))
    thread.start()
    thread = threading.Thread(target=recv_message, args=(clientSocket,))
    thread.start()
    thread = threading.Thread(target=send_suanfa, args=(clientSocket,))
    thread.start()
    TS = 0.2

    for i in range(len(points)):
        while 1:
            if (vehicle.mode.name == "MANUAL") or (vehicle.mode.name == "RTL") or (suanfa == 0):
                c = 0.0
                time.sleep(1)
                continue
            y = points[1] + c + 22
            x = points[0] + c
            ydot = 1
            xdot = 1

            fi = mt.atan2(ydot, xdot)
            R = matrix([[mt.cos(fi), -mt.sin(fi)],
                        [mt.sin(fi), mt.cos(fi)]])
            R = R.T
            p = matrix([[vehicle.location.local_frame.north], [vehicle.location.local_frame.east]])
            pd = matrix([[x], [y]])

            e = R * (p - pd)
            w = mt.atan2(-(e[1][0]), H)
            cmd = fi + w
            cmd = cmd * 180 / mt.pi

            c = TS * vs / r + TS * k * (vs / r) * mt.tanh(e[0][0]) + c - TS * k * mt.tanh(c - cs[0])
            # c = TS*(vs/r) +  c

            uu = mt.sqrt(2) * (vs / r) - 0.5 * mt.tanh(e[0][0])

            YAW_SPEED(cmd, uu, 0)
            time.sleep(TS)
