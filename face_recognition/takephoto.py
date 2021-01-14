import cv2
import os
import time
output_dir = './faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
camera = cv2.VideoCapture(0)
index = 1
while True:
    if (index <= 100):
        print('Taking picture %s' % index)
        time.sleep(0.3)
        success, img = camera.read()
        cv2.imwrite(output_dir+'/'+str(index)+'.jpg', img)
        index += 1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        print('Finish')
        break