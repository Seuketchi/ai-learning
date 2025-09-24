import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import cv2
import mediapipe as mp
import tensorflow as tf
from vggnet import VGGNet
import numpy as np




def inference(image, face_detection, model_1, model_2, emotions):
    H, W, _ = image.shape
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    if results.detections:
        faces = []
        pos = []
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box

            x = int(box.xmin * W)
            y = int(box.ymin * H)
            w = int(box.width * W)
            h = int(box.height * H)

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + w, W)
            y2 = min(y + h, H)

            face = image[y1:y2,x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            faces.append(face)
            pos.append((x1, y1, x2, y2))
    
        x = recognition_preprocessing(faces)

        y_1 = model_1.predict(x)
        y_2 = model_2.predict(x)
        l   = np.argmax(y_1+y_2, axis=1)

        for i in range(len(faces)):
            cv2.rectangle(image, (pos[i][0],pos[i][1]),
                            (pos[i][2],pos[i][3]), emotions[l[i]][1], 2, lineType=cv2.LINE_AA)
            
            cv2.rectangle(image, (pos[i][0],pos[i][1]-20),
                            (pos[i][2]+20,pos[i][1]), emotions[l[i]][1], -1, lineType=cv2.LINE_AA)
            
            cv2.putText(image, f'{emotions[l[i]][0]}', (pos[i][0],pos[i][1]-5),
                            0, 0.6, emotions[l[i]][2], 2, lineType=cv2.LINE_AA)
    
    return image



def resize_face(face):
    x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
    return tf.image.resize(x, (48,48))

def recognition_preprocessing(faces):
    x = tf.convert_to_tensor([resize_face(f) for f in faces])
    return x




if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
        help = "path to the (optional) video file")
    args = vars(ap.parse_args())

    # load the recorded video via the provided path,
    # otherwise 0 (meaning live video stream)
    video = args["video"] if args["video"] is not None else 0 
    cap = cv2.VideoCapture(video)
    
    # Create object for face detection model
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)    


    # Define parameters
    emotions = {
        0: ['Angry', (0,0,255), (255,255,255)],
        1: ['Disgust', (0,102,0), (255,255,255)],
        2: ['Fear', (255,255,153), (0,51,51)],
        3: ['Happy', (153,0,153), (255,255,255)],
        4: ['Sad', (255,0,0), (255,255,255)],
        5: ['Surprise', (0,255,0), (255,255,255)],
        6: ['Neutral', (160,160,160), (255,255,255)]
    }
    num_classes = len(emotions)
    input_shape = (48, 48, 1)
    weights_1 = 'saved_models/vggnet.h5'
    weights_2 = 'saved_models/vggnet_up.h5'
        
        
    # Load the facial emotion recognition models
    # model_1 is a voting classifier trained in original data
    # model_2 is a model trained in upsampled data
    model_1 = VGGNet(input_shape, num_classes, weights_1)
    model_1.load_weights(model_1.checkpoint_path)

    model_2 = VGGNet(input_shape, num_classes, weights_2)
    model_2.load_weights(model_2.checkpoint_path)    

    
    while True:
        success, image = cap.read()
        if success:
            # Perform inference on facial emotions
            result = inference(
                image, 
                face_detection,
                model_1,
                model_2,
                emotions
                )
            cv2.imshow('Facial Emotion', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()



