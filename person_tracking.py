# python3 -m pip install opencv-python
# python3 -m pip install cap-from-youtube
# python3 -m pip install ultralytics

import cv2
from cap_from_youtube import cap_from_youtube
from ultralytics import YOLO
import time  

model = YOLO('yolo11n.pt')

# YouTube video URL to analyze
video_url = 'https://www.youtube.com/watch?v=B4nTR4yWV9g'
cap = cap_from_youtube(video_url,'best')

prev_time = 0  

while True:
    success, img = cap.read()
    if not success or img is None:
        break

    results = model(img, stream=True, classes=[0], conf=0.5)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box around detected person
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Display detection confidence score
            conf = round(float(box.conf[0]), 2)
            cv2.putText(img, f'person {conf}', (x1, y1-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

    # FPS calculations
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # FPS display
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('person_tracking', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()