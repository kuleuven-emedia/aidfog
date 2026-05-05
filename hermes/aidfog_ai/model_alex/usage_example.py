"""
usage_example.py — minimal load + infer demo for model_alex.

Input contract:
- Sample rate  : 60 Hz
- Array shape  : (T, 30)  — T timesteps, 30 channels
                 5 IMUs × 6 axes: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
                 IMU order: Pelvic, Left_tibia, Left_talus, Right_tibia, Right_talus
- Normalisation: None required — handled internally via rolling window mean

Model selection (LOSOCV):
- One .pt file per subject lives in model_alex/weights/subject{id}.pt
- Each model was trained with that subject held out, so you MUST pass the
  correct subject ID. This is not optional — using the wrong model invalidates
  predictions for that subject.

Production postprocessing:
- thresh       : 0.5
- enter_thresh : 20
- exit_thresh  : 5
"""

import numpy as np
from model_alex.streaming import predict_streaming

# Simulate 10 seconds of raw IMU data at 60 Hz, 30 channels
T = 60 * 10   # 600 samples
imu_window = np.random.randn(T, 30).astype(np.float32)

# Subject ID selects weights/subject010.pt
# Replace '010' with the actual subject being evaluated
subject = '010'

prob_smoothed, binary = predict_streaming(
    imu_window,
    subject=subject,
    enter_thresh=20,
    exit_thresh=5,
)

print("probas shape:", prob_smoothed.shape)   # (600,)
print("preds  shape:", binary.shape)          # (600,)
print("prob sample :", prob_smoothed[:5])
print("pred sample :", binary[:5])
