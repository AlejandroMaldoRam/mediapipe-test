# Script for applying media pipe in videos

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2

 # Configure model
model_path = "D:\Code\mediapipe-test\models\pose_landmarker_heavy.task"
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode



# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))
    cv2.imshow("live", output_image)

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
   
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

if __name__ == '__main__':
    options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)
    with PoseLandmarker.create_from_options(options) as landmarker:
        # Open Video capture
        cap = cv2.VideoCapture(0)

        i = 0
        period = 25
        while cap.isOpened():
            frame_timestamp_ms = i*period
            ret, img = cap.read()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            #landmarker.detect_async(mp_image, frame_timestamp_ms)
            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            #detection_result = detector.detect(mp_image)
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
        
            cv2.imshow("resultado",annotated_image)
            #cv2.imshow("imagen", img)
            k = cv2.waitKey(25)
            if (k & 0xFF)==ord('q'):
                break
            i+=1
        cap.release()
        cv2.destroyAllWindows()