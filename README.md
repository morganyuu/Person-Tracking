# Person-Tracking
Real-time person detection on YouTube videos using YOLO and OpenCV, with FPS display.

--

## Project Overview

This project uses a YOLO (Ultralytics) object detection model to identify people in YouTube videos in real time, displaying bounding boxes with confidence scores. An FPS (frames per second) counter is included to demonstrate how efficiently the model processes video frames.

Through testing, the lightweight `yolo11n.pt` model achieved the best performance, averaging around 20–21 FPS. Larger YOLO models (such as the “x” variants) were significantly slower, running at approximately 2 FPS. The model can be easily changed by replacing the model filename with any other supported YOLO model.

The `cap_from_youtube` library is used to convert a YouTube video URL into an OpenCV-compatible video capture object, allowing the video to be processed as if it were a local file. The test URL used in this project is *“What Song Are You Listening To? TOKYO, JAPAN”* by JESSEOGN, but it can be replaced with any other YouTube video link.

YOLO also supports detecting a wide range of objects from the COCO (Common Objects in Context) dataset. This code can be modified to detect any of the 80 COCO classes by specifying the desired class indices in the detection results (for example: `0` for person, `2` for car, `41` for cup, `47` for apple).
