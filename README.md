# mediapipe-test
Repository for testing MediaPipe framework.

# Compilation

pyinstaller --onefile --windowed --add-data "Diapositiva1.PNG;." --add-data "Diapositiva2.PNG;." --add-data "Diapositiva3.PNG;." --add-data "Diapositiva4.PNG;." --add-data "Diapositiva5.PNG;." --add-data "Diapositiva6.PNG;." --add-data "Diapositiva7.PNG;." --add-data "Diapositiva8.PNG;." --add-data "Diapositiva9.PNG;." --add-data "Diapositiva10.PNG;." --add-data "efficientdet_lite0_int8.tflite;." fd-slideshow.py
