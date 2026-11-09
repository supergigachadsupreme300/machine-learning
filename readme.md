
# Hand Gesture Recognition Dataset

## Project
Hand gesture recognition for computer control.

## Classes
- MOUSE_UP
- MOUSE_DOWN
- MOUSE_LEFT
- MOUSE_RIGHT
- SCROLL_UP
- SCROLL_DOWN
- SCROLL_LEFT
- SCROLL_RIGHT
- CLICK
- ZOOM_IN
- ZOOM_OUT
- NEUTRAL

## Data Collection
Images collected using webcam and OpenCV.

## Dataset Split
- Train: 70%
- Validation: 15%
- Test: 15%

## Dataset Structure
- data/raw: original collected images
- data/split/train: training data
- data/split/val: validation data
- data/split/test: testing data

## Notes
Original images are preserved in data/raw.



LƯU Ý ở dataset : Code đang lật ảnh webcam bằng cv2.flip(frame, 1) để thao tác giống gương.  thống nhất cách lật này trong suốt quá trình thu thập, không lúc lật lúc không.

