import cv2

video_path = "data/dashcam.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    print("Video opened succesfully.")

cap.release()
