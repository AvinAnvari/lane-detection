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
    cv2.imshow("Dashcam Feed",frame)
    key = cv2.waitKey(25)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
