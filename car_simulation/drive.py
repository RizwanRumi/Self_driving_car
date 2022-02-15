import socketio
import eventlet
import numpy as np
import cv2
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image

sio = socketio.Server()

app = Flask(__name__)
speed_limit = 10
def img_preprocess(img):
  # delete irrelivant features from image
  img = img[60:135, :,: ]
  # change color space to YUV because of using NVIDIA model architecture
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  # apply gaussian blur for smoothing and reducing noise
  img = cv2.GaussianBlur(img, (3, 3), 0)
  # for faster computation reduce the image size
  img = cv2.resize(img, (200, 66)) # nvidia input image size
  # finally apply normalization
  img = img/255
  return img

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0,0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
'''
@app.route('/home')
def greeting():
    return 'welcome to the self driving car simulator'
'''

if __name__ == '__main__':
    #app.run(port=5000)
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)