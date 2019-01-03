import numpy as np
import cv2
from imageai.Detection import ObjectDetection
import os
import tensorflow as tf
import tarfile
import six.moves.urllib as urllib
import pandas as pd
import time
from _slack_msg import *
from datetime import datetime
import os

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as pyplot
from PIL import Image

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('D:/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

null_list = ('surfboard', 'bench', 'chair', 'sink')

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
	file_name = os.path.basename(file.name)
	if 'frozen_inference_graph.pb' in file_name:
		tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

video_ip = # Add IP for video 
cap = cv2.VideoCapture(video_ip)

df = pd.DataFrame(columns=['Objects', 'Count', 'Timer', 'New', 'Picture'])
df['Count'] = [0, 0, 0, 0, 0]
df['Timer'] = [0, 0, 0, 0, 0]
df['New'] = [0, 0, 0, 0, 0]
df['Picture'] = [0, 0, 0, 0, 0]

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		while(True):
			ct = float(datetime.now().strftime('%H')) 

			ret, image_np = cap.read()
			if (ct >= 17) and (ct <= 6):
				image_np = increase_brightness(image_np, value=50)
			image_np_expanded = np.expand_dims(image_np, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			#Detection
			(boxes, scores, classes, num_detections) = sess.run(
				[boxes, scores, classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})

			for index, row in df.iterrows():
				if df.at[index, 'Timer'] != 0:
					if df.at[index, 'Timer']+120 < int(time.time()):
						df.at[index, 'Timer'] = 0

			#Visualization of results
			vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				category_index,
				use_normalized_coordinates=True,
				line_thickness=3)

			cv2.imshow('object detection', cv2.resize(image_np, (1000,600)))


			#objects
			_object = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.6]
			if _object:
				df.at[0, 'Objects'] = _object[0]['name']
				if df.at[0, 'Timer'] == 0:
					df.at[0, 'Count'] = df.at[0, 'Count'] + 1
					df.at[0, 'New'] = 1
					print(str(datetime.now().strftime("%x %X")) + ': ' + str(_object[0]['name']) + ' at the door.')
				if str(_object[0]['name']) == 'person':
					n = 10
				else:
					n = 8

				if (ct > 17) and (ct < 6):
					n = int(n/3)

				if (df.at[0, 'Count'] > n) and (str(_object[0]['name']) not in null_list):
					msg = str(_object[0]['name']) + ' at the door.'
					print(msg)
					slack_message(msg, 'camera', 'Front Door Camera')
					img_file = 'D:/security_cam/' + str(int(time.time())) + '.jpg'
					if (ct > 1) and (ct < 14):
						image_np_110 = increase_brightness(image_np, value=110)
						img_file_110 = 'D:/security_cam/' + str(int(time.time())) + '_110.jpg'
						cv2.imwrite(img_file_110, cv2.resize(image_np_110, (1000,600)))
						print('Snapshot saved to ' + str(img_file_110))
					cv2.imwrite(img_file, cv2.resize(image_np, (1000,600)))
					print('Snapshot saved to ' + str(img_file))
					slack_image(img_file, 'camera')
					df.at[0, 'Timer'] = int(time.time())
					df.at[0, 'Count'] = 0

				if (df.at[0, 'Picture'] >= 2) and (str(_object[0]['name']) not in null_list):
					today = str(datetime.now().strftime("%Y-%b-%d"))
					path = 'D:/security_cam/' + str(today)
					try:
						os.mkdir(path)
					except:
						pass
					img_file_a = str(path) + '/front_' + str(int(time.time())) + '_' + str(_object[0]['name']) + '.jpg'
					print(str(img_file_a) + ' saving.')
					cv2.imwrite(img_file_a, cv2.resize(image_np, (1000,600)))
					print(str(img_file_a) + ' saved.')
					df.at[0, 'Picture'] = 0
				else:
					df.at[0, 'Picture'] = df.at[0, 'Picture'] + 1

			elif df.at[0, 'New'] != 0:
				df.at[0, 'New'] = 0
				df.at[0, 'Count'] = 0


			if cv2.waitKey(25) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

cap.release()
cv2.destroyAllWindows()

