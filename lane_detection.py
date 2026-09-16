import cv2
import numpy as np

# --- Config / tunable constants -------------------------------------------

VIDEO_PATH = "data/dashcam.mp4"
FRAME_SIZE = (960, 540)          # (width, height) frames are resized to before processing

BLUR_KERNEL = (7, 7)             # Gaussian blur kernel size

CANNY_LOW = 60                   # Canny edge detection thresholds
CANNY_HIGH = 160

# Region-of-interest trapezoid, tuned for this camera's mounting angle at
# FRAME_SIZE resolution. A different dashcam/angle will need new points.
ROI_VERTICES = np.array([[
    (150, 300),
    (785, 300),
    (550, 150),
    (400, 150),
]])

# Hough transform params (rho, theta, threshold, minLineLength, maxLineGap)
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 2
HOUGH_MIN_LINE_LENGTH = 2
HOUGH_MAX_LINE_GAP = 7

SLOPE_THRESHOLD = 0.3            # lines flatter than this (near-horizontal) are discarded as noise

# y-coordinates each averaged lane line is extrapolated between
LINE_Y_BOTTOM = 350
LINE_Y_TOP = 170

LEFT_LINE_COLOR = (255, 0, 0)    # BGR
RIGHT_LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 10


# --- Pipeline steps ---------------------------------------------------------

def preprocess_frame(frame):
    """Resize, grayscale, blur, and run Canny edge detection on one frame."""
    frame = cv2.resize(frame, FRAME_SIZE)
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, BLUR_KERNEL, 0)
    edge = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    return frame, edge


def region_of_interest(edge):
    """Mask out everything outside the road-shaped ROI trapezoid."""
    mask = np.zeros(edge.shape, dtype=np.uint8)
    filled = cv2.fillPoly(mask, ROI_VERTICES, 255)
    return cv2.bitwise_and(filled, edge)


def detect_line_segments(bitwised):
    """Run the probabilistic Hough transform to find candidate line segments."""
    return cv2.HoughLinesP(
        bitwised,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        HOUGH_MIN_LINE_LENGTH,
        HOUGH_MAX_LINE_GAP,
    )


def average_slope_intercept(lines):
    """
    Split detected segments into left/right groups by slope sign, discard
    near-horizontal noise, and average each group into a single (slope, intercept).

    Returns (left_avg, right_avg), where each is a (slope, intercept) tuple
    or None if no segments were found on that side.
    """
    left_lines_slopes = []
    right_lines_slopes = []
    left_lines_intercepts = []
    right_lines_intercepts = []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
        else:
            continue

        intercept = y1 - slope * x1

        if slope < -SLOPE_THRESHOLD:  # Filtering out left and right and horizontal noise
            left_lines_slopes.append(slope)
            left_lines_intercepts.append(intercept)
        elif slope > SLOPE_THRESHOLD:
            right_lines_slopes.append(slope)
            right_lines_intercepts.append(intercept)

    left_avg = None
    if len(left_lines_slopes) > 0:
        left_avg = (
            sum(left_lines_slopes) / len(left_lines_slopes),
            sum(left_lines_intercepts) / len(left_lines_intercepts),
        )

    right_avg = None
    if len(right_lines_slopes) > 0:
        right_avg = (
            sum(right_lines_slopes) / len(right_lines_slopes),
            sum(right_lines_intercepts) / len(right_lines_intercepts),
        )

    return left_avg, right_avg


def make_line_points(slope_intercept, y1, y2):
    """Convert a (slope, intercept) pair into (x1, y1), (x2, y2) pixel endpoints."""
    slope, intercept = slope_intercept
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, y1), (x2, y2)


def draw_lane_lines(frame, left_avg, right_avg):
    """Draw the extrapolated left/right lane lines onto the frame in place."""
    if left_avg is not None:
        pt1, pt2 = make_line_points(left_avg, LINE_Y_BOTTOM, LINE_Y_TOP)
        cv2.line(frame, pt1, pt2, LEFT_LINE_COLOR, LINE_THICKNESS)

    if right_avg is not None:
        pt1, pt2 = make_line_points(right_avg, LINE_Y_BOTTOM, LINE_Y_TOP)
        cv2.line(frame, pt1, pt2, RIGHT_LINE_COLOR, LINE_THICKNESS)


def process_frame(frame):
    """Run the full pipeline on one frame and return the frame with lanes drawn."""
    frame, edge = preprocess_frame(frame)
    bitwised = region_of_interest(edge)
    lines = detect_line_segments(bitwised)
    left_avg, right_avg = average_slope_intercept(lines)
    draw_lane_lines(frame, left_avg, right_avg)
    return frame


# --- Entry point -------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = process_frame(frame)
        cv2.imshow("Dashcam Feed", output)

        key = cv2.waitKey(25)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()