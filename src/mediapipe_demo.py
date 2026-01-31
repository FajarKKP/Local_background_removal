import cv2
import numpy as np
import time 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions, RunningMode

model_path = "D:/Passion/Inventor/Code/Local_background_removal/model/deeplab_v3.tflite"

BaseOptions = mp.tasks.BaseOptions
options = ImageSegmenterOptions(
    base_options = BaseOptions(model_asset_path = model_path),
    running_mode = RunningMode.VIDEO,
    output_confidence_masks = True
)

segmenter = ImageSegmenter.create_from_options(options)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_mask = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    ts = int(time.time() * 1000)

    result = segmenter.segment_for_video(mp_image, ts)

    mask = np.array(result.confidence_masks[0].numpy_view(), dtype=np.float32)

    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    if prev_mask is not None:
        alpha = 0.7
        mask = alpha * mask + (1 - alpha) * prev_mask
    prev_mask = mask

    mask = np.clip(mask, 0, 1)
    mask_3c = np.stack([mask] * 3, axis=-1)

    blurred_bg = cv2.GaussianBlur(frame, (55,55), 0)

    # Blur background
    output = (mask_3c * blurred_bg + (1 - mask_3c) * frame).astype(np.uint8)

    # # Blur foreground
    # output = (mask_3c * frame + (1 - mask_3c) * blurred_bg).astype(np.uint8)


    cv2.imshow("Background Blur (test with mediapipe)", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()