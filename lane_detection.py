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
    edge = cv2.Canny(blur, 30, 100)
    points = np.array([[
    (150, 300),
    (785, 300),
    (550, 150),
    (400, 150)]])
    mask = np.zeros(edge.shape, dtype=np.uint8)
    filled = cv2.fillPoly(mask, points,255)
    bitwised = cv2.bitwise_and(filled, edge)
    # test_lines = cv2.polylines(frame,points, True, (0,0,255), 5)
    lines = cv2.HoughLinesP(bitwised, 1, np.pi/180, 2, 2, 7)

    left_lines_slopes = []
    right_lines_slopes = []
    left_lines_intercepts = []
    right_lines_intercepts = []
    if lines is not None:
        print("num lines found:", len(lines))
        for line in lines:
            x1, y1, x2, y2 = line
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
            else:
                    continue
            intercept = y1 - slope * x1
            if slope < 0:
                    left_lines_slopes.append(slope)
                    left_lines_intercepts.append(intercept)
            else:
                right_lines_slopes.append(slope)
                right_lines_intercepts.append(intercept)
            cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 10)

        if len(left_lines_slopes) > 0:
            left_avg_slope = sum(left_lines_slopes)/len(left_lines_slopes)
            left_avg_intercept = sum(left_lines_intercepts)/len(left_lines_intercepts)
        if len(right_lines_slopes) > 0:
            right_avg_slope = sum(right_lines_slopes)/len(right_lines_slopes)
            right_avg_intercept = sum(right_lines_intercepts)/len(right_lines_intercepts)
        print(right_avg_slope, right_avg_intercept, left_avg_slope, left_avg_intercept)
    else:
        print("num lines found: 0")

    cv2.imshow("Dashcam Feed",frame)
    key = cv2.waitKey(25)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
