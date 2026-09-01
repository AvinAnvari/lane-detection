import cv2
import numpy as np

video_path = "data/dashcam.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (960, 540))
    
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (7,7), 0)
    edge = cv2.Canny(blur, 60, 160)
    points = np.array([[
    (150, 300),
    (785, 300),
    (550, 150),
    (400, 150)]])
    mask = np.zeros(edge.shape, dtype=np.uint8)
    filled = cv2.fillPoly(mask, points,255)
    bitwised = cv2.bitwise_and(filled, edge)
    # test_lines = cv2.polylines(frame,points, True, (0,0,255), 5)
    lines = cv2.HoughLinesP(bitwised, 0.5, np.pi/180, 1, 2, 1)

    if lines is not None:
        for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 10)

    cv2.imshow("Dashcam Feed",frame)
    key = cv2.waitKey(25)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
