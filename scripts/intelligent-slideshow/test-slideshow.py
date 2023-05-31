# Script for testing the idea of an intelligente slideshow that detects when a person is
# looking to a camera.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import math
import numpy as np
import cv2
from typing import Tuple, Union

 # Configure model
model_path = "../../models/blaze_face_short_range.tflite"

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

slides = ["../../slides/Diapositiva1.PNG","../../slides/Diapositiva2.PNG",
          "../../slides/Diapositiva3.PNG","../../slides/Diapositiva4.PNG",
          "../../slides/Diapositiva5.PNG","../../slides/Diapositiva6.PNG"]

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image

if __name__ == '__main__':
    # Load slides
    slides_imgs = []
    for f in slides:
       img = cv2.imread(f)
       slides_imgs.append(img)
    
    options = FaceDetectorOptions(base_options=BaseOptions(model_asset_path=model_path),running_mode=VisionRunningMode.VIDEO)
    with FaceDetector.create_from_options(options) as detector:
        # Open Video capture
        cap = cv2.VideoCapture(0)

        i = 0
        period = 25
        slide_time = [40,80,120,160,200]
        face_counter = 0
        while cap.isOpened():
            frame_timestamp_ms = i*period
            ret, img = cap.read()
            area_img = img.shape[0]*img.shape[1]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            #landmarker.detect_async(mp_image, frame_timestamp_ms)
            face_detector_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        
            if len(face_detector_result.detections)>0:
                #print("{} caras detectadas.".format(len(face_detector_result.detections)))
                for j, detection in enumerate(face_detector_result.detections):
                    bbox = detection.bounding_box
                    area = bbox.width*bbox.height
                    #print("Area cara {}: {}".format(j, 100*area/area_img))
                    if 100*area/area_img>3.0:
                       face_counter += 1
                    else:
                       face_counter = 0
            else:
               face_counter = 0
            annotated_image = visualize(mp_image.numpy_view(), face_detector_result)
            cv2.imshow("resultado",annotated_image)
            if face_counter<slide_time[0]: #aproxx 2 seconds
                cv2.imshow("slides", slides_imgs[0])
            elif face_counter<slide_time[1]:
                cv2.imshow("slides", slides_imgs[2])
            elif face_counter<slide_time[2]:
                cv2.imshow("slides", slides_imgs[3])
            elif face_counter<slide_time[3]:
                cv2.imshow("slides", slides_imgs[4])
            elif face_counter<slide_time[4]:
                cv2.imshow("slides", slides_imgs[5]) 
            else:
                cv2.imshow("slides", slides_imgs[0])
                face_counter = 0
            k = cv2.waitKey(25)
            if (k & 0xFF)==ord('q'):
                break
            i+=1
        cap.release()
        cv2.destroyAllWindows()