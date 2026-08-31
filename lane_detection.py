import cv2

video_path = "data/dashcam.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (7,7), 0)
    edge = cv2.Canny(blur, 80, 160)
    cv2.imshow("Dashcam Feed",edge)
    key = cv2.waitKey(25)

    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
