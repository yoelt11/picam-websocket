import websockets
import cv2
import asyncio
from queue import Queue
from threading import Thread, Event
import sys
import time


async def sendImage():
	''' 
		sends image over websocket
	'''
	while True:
		async with websockets.connect('ws://10.0.0.1:6000') as websocket:
			while True:
				try:
					image = image_queue.get()
					image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
					print("sending image bytes: ", len(image_bytes))
					await websocket.send(image_bytes)
				except websockets.ConnectionClosed:
					break

def getImage():
	
	video_feed = cv2.VideoCapture(0)
	# video_feed.set(cv2.CAP_PROP_FPS, 20)
	
	while True:
		ret, frame = video_feed.read()
		image_queue.put(frame)



if __name__ == '__main__':

	image_queue = Queue() 

	address = sys.argv[1]

	camera_thread = Thread(target=getImage)
	camera_thread.start()

	asyncio.run(sendImage())

	camera_thread.join()