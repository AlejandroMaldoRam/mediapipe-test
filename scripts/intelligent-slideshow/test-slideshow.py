# Scrip for detecting objects using real time feed.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

MODEL_PATH = 'D:/Code/mediapipe-test/models/efficientdet_lite0_int8.tflite'

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
CLASSES_OF_INTEREST = ['person']

SLIDES = ["D:\code\mediapipe-test\slides\MOBF2023_v2\Diapositiva1.PNG",
              "D:\code\mediapipe-test\slides\MOBF2023_v2\Diapositiva2.PNG",
              "D:\code\mediapipe-test\slides\MOBF2023_v2\Diapositiva3.PNG",
              "D:\code\mediapipe-test\slides\MOBF2023_v2\Diapositiva4.PNG",
              "D:\code\mediapipe-test\slides\MOBF2023_v2\Diapositiva5.PNG",
              "D:\code\mediapipe-test\slides\MOBF2023_v2\Diapositiva6.PNG",
              "D:\code\mediapipe-test\slides\MOBF2023_v2\Diapositiva7.PNG",
              "D:\code\mediapipe-test\slides\MOBF2023_v2\Diapositiva8.PNG",
              "D:\code\mediapipe-test\slides\MOBF2023_v2\Diapositiva9.PNG"]

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

def add_img_to_corner(img1, img2):
  "This function returns the result of putting the img2 in the lower right corner of img1"
  img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)
  h,w,c = img1.shape
  h2,w2,c3 = img2.shape
  img1[h-h2:,w-w2:,:] = img2
  return img1


if __name__ == '__main__':
    slides_imgs = []
    for f in SLIDES:
        img = cv2.imread(f)
        slides_imgs.append(img)

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
        detections_counter = 0
        undetected_counter = 0

        while cap.isOpened():
            frame_timestamp_ms = i*period
            ret, img = cap.read()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
            annotated_image = visualize(mp_image.numpy_view(), detection_result)
            if len(detection_result.detections)>0:
              detections_counter += 1
              undetected_counter = 0
            else:
              undetected_counter += 1
              if undetected_counter>3:
                detections_counter = 0
            
            if detections_counter<30:
              annotated_img2 = add_img_to_corner(slides_imgs[0], annotated_image)
            elif detections_counter<60:
              annotated_img2 = add_img_to_corner(slides_imgs[1], annotated_image)
            elif detections_counter<90:
              annotated_img2 = add_img_to_corner(slides_imgs[3], annotated_image)
            elif detections_counter<120:
              annotated_img2 = add_img_to_corner(slides_imgs[4], annotated_image)
            elif detections_counter<150:
              annotated_img2 = add_img_to_corner(slides_imgs[5], annotated_image)
            elif detections_counter<180:
              annotated_img2 = add_img_to_corner(slides_imgs[6], annotated_image)
            elif detections_counter<210:
              annotated_img2 = add_img_to_corner(slides_imgs[7], annotated_image)
            elif detections_counter<240:
              annotated_img2 = add_img_to_corner(slides_imgs[8], annotated_image)
            else:
              annotated_img2 = add_img_to_corner(slides_imgs[0], annotated_image)
            annotated_img2 = cv2.resize(annotated_img2, (0,0), fx=0.5, fy=0.5)
            cv2.imshow("Presentación", annotated_img2)
            k = cv2.waitKey(period)
            if (k & 0xFF)==ord('q'):
                break
            i+=1
        cap.release()
        cv2.destroyAllWindows()