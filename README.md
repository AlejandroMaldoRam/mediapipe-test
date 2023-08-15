# mediapipe-test
Repository for testing MediaPipe framework.

# Compilation

pyinstaller --onefile --windowed --add-data "Diapositiva1.PNG;." --add-data "Diapositiva2.PNG;." --add-data "Diapositiva3.PNG;." --add-data "Diapositiva4.PNG;." --add-data "Diapositiva5.PNG;." --add-data "Diapositiva6.PNG;." --add-data "Diapositiva7.PNG;." --add-data "Diapositiva8.PNG;." --add-data "Diapositiva9.PNG;." --add-data "Diapositiva10.PNG;." --add-data "efficientdet_lite0_int8.tflite;." fd-slideshow.py

# Usage

1. Open Anaconda PowerShell.
2. cd .\code\mediapipe-test\scripts\intelligent-slideshow
3. conda activate mp-env
4. python .\test-slideshow.py
5. You can stop the program pressing 'q'.

* You can change the image address in lines 17-25 in test-slideshow.py.
