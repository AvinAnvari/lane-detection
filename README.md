# Lane Detection

A computer vision pipeline that detects lane lines in dashcam footage using classical image processing techniques — built from scratch with OpenCV and NumPy, without any pre-trained models or deep learning.

## What it does

Given a dashcam video, the pipeline:

1. Reads the video frame by frame
2. Converts each frame to grayscale and applies Gaussian blur to reduce noise
3. Runs Canny edge detection to find high-contrast boundaries (like lane markings against asphalt)
4. Masks out everything except a region of interest (ROI) — a trapezoid shaped like the road ahead — to ignore irrelevant edges (sky, trees, dashboard, etc.)
5. Applies the Probabilistic Hough Line Transform to find straight line segments within that masked region
6. Splits detected segments into "left lane" and "right lane" groups based on their slope, filters out near-horizontal noise, and averages each group into a single representative line
7. Extrapolates each averaged line across the full height of the ROI and draws it on the original video

## Pipeline overview

```
Raw frame → Grayscale → Gaussian Blur → Canny Edge Detection
    → ROI Mask → Hough Line Transform
    → Slope-based filtering & sorting (left / right)
    → Averaging & extrapolation → Final overlay
```

## Tech stack

- Python 3
- OpenCV (`opencv-python`)
- NumPy

## Running it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python lane_detection.py
```

Press `q` at any time to quit the video window early.

## Known limitations

This is a deliberately from-scratch, classical CV approach (no deep learning), and it has some real, known limitations that would need more advanced techniques to solve:

- **Static region of interest**: the ROI trapezoid is currently hardcoded (or percentage-based, calibrated for one camera angle), so it doesn't automatically adapt to different camera mounting positions or fields of view. A different dashcam video may require re-tuning the ROI coordinates.
- **Struggles on curves**: since the ROI and lane-fitting logic assume roughly straight lane lines, sharp curves cause the mask and detected lines to fall out of alignment with the actual road.
- **No frame-to-frame smoothing**: each frame is processed independently, so the detected lines can jitter slightly between frames rather than moving smoothly.
- **Sensitive to lighting/contrast**: Canny edge detection thresholds are tuned for the test footage's lighting conditions and may need adjustment for videos shot in different weather or lighting.

## Possible future improvements

- Percentage-based or camera-calibration-derived ROI that generalizes across different videos/camera setups
- Polynomial curve fitting (instead of straight-line extrapolation) to handle curved roads
- Temporal smoothing across frames (e.g., exponential moving average of slope/intercept) to reduce jitter
- Outlier rejection before averaging, to reduce the influence of noisy/incorrect line detections

## Why classical CV instead of deep learning?

This project was built to deeply understand the fundamentals of image processing and computer vision — edge detection, region masking, and line fitting — rather than relying on a pre-trained model as a black box. Every function and parameter choice was tuned and understood individually, including significant debugging around Hough Transform parameter tuning for short, dashed lane markings at lower resolutions.
