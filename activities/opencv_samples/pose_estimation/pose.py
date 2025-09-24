# Usage:
# $ python pose.py
# $ python pose.py --video video/adrian_face.mov


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import cv2
import mediapipe as mp



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
        help = "path to the (optional) video file")
    args = vars(ap.parse_args())

    # load the recorded video via the provided path,
    # otherwise 0 (meaning live video stream)
    video = args["video"] if args["video"] is not None else 0 
    cap = cv2.VideoCapture(video)

    # Load Mediapipe's pose model
    mp_pose = mp.solutions.pose

    # Define parameters
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles


    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            success, image = cap.read()
            if success:
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Perform prediction with pose
                results = pose.process(image)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    
    cap.release()   
    cv2.destroyAllWindows()             
