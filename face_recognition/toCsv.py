import os
import numpy as np
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

def tocsv(path):
    if not os.path.exists(path+'_csv'):
        os.makedirs(path+'_csv')
    for dir in os.listdir(path):
        dirpath=os.path.join(path,dir)
        copydir=os.path.join(path + '_csv',dir)
        if os.path.isdir(dirpath):
            if not os.path.exists(copydir):
                os.makedirs(copydir)
            count=0
            for img in image_files_in_folder(dirpath):
                face = face_recognition.load_image_file(img)
                face_bounding_boxes = face_recognition.face_locations(face)
                if len(face_bounding_boxes) == 1:
                    print(img)
                    count=count+1
                    face_enc = face_recognition.face_encodings(face)[0]
                    csvname=os.path.join(copydir,str(count))
                    np.savetxt(csvname+'.csv', face_enc, delimiter=',')

tocsv('./test_data')