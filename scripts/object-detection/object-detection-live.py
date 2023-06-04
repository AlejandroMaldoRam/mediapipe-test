# Scrip for detecting objects using real time feed.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

MODEL_PATH = 'D:\Code\mediapipe-test\models\efficientdet_lite0_int8.tflite'

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
CLASSES_OF_INTEREST = ['person']


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

if __name__ == '__main__':
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        max_results=5,
        running_mode=VisionRunningMode.VIDEO,
        category_allowlist=CLASSES_OF_INTEREST,
        score_threshold=0.75)
    
    with ObjectDetector.create_from_options(options) as detector:
        # Open Video capture
        cap = cv2.VideoCapture(0)
        i = 0
        period = 25
        while cap.isOpened():
            frame_timestamp_ms = i*period
            ret, img = cap.read()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
            print("Detections: ", len(detection_result.detections))
            annotated_image = visualize(mp_image.numpy_view(), detection_result)
            print(detection_result)
            cv2.imshow("Entrada", annotated_image)
            k = cv2.waitKey(period)
            if (k & 0xFF)==ord('q'):
                break
            i+=1
        cap.release()
        cv2.destroyAllWindows()