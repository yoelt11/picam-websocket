import asyncio
import json
import yaml
import websockets as websockets
from PIL import Image
import cv2
import numpy as np
from queue import Queue
from PoseClassification import PoseDetectionEngine
import torch


import time
from threading import Thread, Event

async def getImageFromWebsocket(websocket):
    """ Connects to Java via Websocket and gets images and distributes them two the object detection task and
        Pose Classification task
    """
    print('Computer-vision server Started')
    while True:
        async for image_bytes in websocket:

            if image_bytes != None:
                # Decode Image
                decoded_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
                frame = np.array(Image.fromarray(decoded_image))
                cv2.imshow('Camera Feed', decoded_image)
                input_queue.put(frame)
                #await asyncio.sleep(1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


def localCamera():

    print("Getting image locally")

    video_feed = cv2.VideoCapture(0)
    video_feed.set(cv2.CAP_PROP_FPS, 20)
    
    while(True):

        ret, frame = video_feed.read()

        cv2.imshow('Camera Feed', frame)

        # Proceed if image_byte != empty
        if frame.any() != None:
            input_queue.put(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def objectDetection(image):
    """Runs Object Detections and puts result in the output queue as a json string"""
    if image is not None:
        detectedObjects = objectDetectionEngine.detect(image)
        jsonResponse = objectDetectionEngine.generateJsonResponse(detectedObjects)
        output_queue.put(jsonResponse)


def poseDetection(image):
    """Runs Pose Classification and puts result in the output queue as a json string"""

    if image is not None:
        pose_raw_output = poseDetectionEngine.detect(image)

    return pose_raw_output


def recordData(nvideos):
    """ Parses the frames according to the fps and sends it to either Object Detection or Activity Recognition,
     option variable can be adjusted in the json file"""
    name = "standing"
    PATH = '/home/etorres/Documents/in-work/datasets/activity-recognition/new_raw/'
    frame_count = 0
    T = 30
    X = torch.zeros([nvideos, T, 17, 3])
    # Frame rate
    B = 0 
    pause = 10
    video_count = 0
    #out = cv2.VideoWriter(PATH + name + '_' + str(video_count) + '.avi', 0, 5.0, (640,480))
    while(True):
        if config_param['VIDEO_RECORD']['STATUS'] == True:
            frame = input_queue.get()
            
            if frame_count > pause and frame_count < 40:
                print(f'recording: {frame_count}')
                #out.write(frame)
            elif frame_count > 40:
                frame_count = 0
                video_count += 1
                #out = cv2.VideoWriter(PATH + name + '_' + str(video_count) + '.avi', 0, 5.0, (640,480))
            frame_count += 1
            if video_count == nvideos:
                break
        # frame_count = frame_count + 1
        # T = 0
        # if frame_count < 40 and frame_count >=10:
        #     print(f'recording frame: {frame_count}')
        #     frame = input_queue.get()
        #     if config_param['POSE_DETECTION']['STATUS'] == True:
        #         X[B,T,:,:] = torch.tensor(poseDetection(frame))
        #         T += 1
        #     if config_param['VIDEO_RECORD']['STATUS'] == True:
        #         previous_detected_activity, activity_raw_output, pose_raw_output = poseClassification(frame, previous_detected_activity, activity_raw_output, pose_raw_output, False)
        # elif frame_count > 40:
        #     frame_count = 0
        #     B += 1
        

def load_yaml(PATH='./config.yaml'):
    """
        Reads yaml configuration script
    """
    stream = open(PATH, 'r')
    dictionary = yaml.safe_load(stream)
    
    return dictionary


if __name__ == '__main__':

    # Input and Output queue
    input_queue = Queue()  # queue for the incomming images
    output_queue = Queue()  # queue holding response

    # Set Options
    config_param = load_yaml()

    #nvidios
    nvideos = 100

    # Set Port
    PORT = config_param['CONNECTION']['PORT']
    HOST = config_param['CONNECTION']['HOST']

    # Start Activity Recognition
    if config_param['POSE_DETECTION']['STATUS']:
        # Start Pose Detection Engine
        POSE_MODEL = config_param['POSE_DETECTION']['MODEL_DIR']
        poseDetectionEngine = PoseDetectionEngine(POSE_MODEL)

    # Starts Prorcessing Thread
    record_data = Thread(target=recordData, args=[nvideos])
    record_data.start()

    if config_param['LOCAL_CAMARA']:
        # Starts camera locally
        camera = Thread(target=localCamera)
        camera.start()
    else:
        # Starts Server
        async def server():
            async with websockets.serve(getImageFromWebsocket, HOST, PORT, ping_timeout=None):
                await asyncio.Future()

        asyncio.run(server())

    computer_vision.join()
    camera.join()
