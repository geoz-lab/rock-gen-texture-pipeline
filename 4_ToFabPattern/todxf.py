import cv2
import numpy as np
import ezdxf

# --- SETTINGS ---
input_image = './rock_cropped.png'
output_dxf = 'rock_background.dxf'
min_area = 5        # minimum contour area to keep

# --- STEP 1: Load and resize image ---
print("Loading and resizing image to 5001 x 2001...")
img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load image: {input_image}")

# Resize image to fixed size
resized_img = cv2.resize(img, (5001, 2001), interpolation=cv2.INTER_NEAREST)
height, width = resized_img.shape  # now 5001, 2001

# --- STEP 2: Apply Otsu thresholding ---
print("Applying thresholding...")
_, binary = cv2.threshold(resized_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- STEP 3: Invert binary to get background as foreground ---
binary_inv = cv2.bitwise_not(binary)

# --- STEP 4: Find contours ---
print("Finding contours for background region...")
contours, hierarchy = cv2.findContours(binary_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# --- STEP 5: Calculate global minimum X, Y for shifting to (0,0) ---
all_points = np.vstack([cnt.reshape(-1, 2) for cnt in contours])
min_x = np.min(all_points[:, 0])
min_y = np.min(all_points[:, 1])

# --- STEP 6: Prepare DXF ---
doc = ezdxf.new()
msp = doc.modelspace()
outer_count = 0
hole_count = 0

# Final scale factor: 4x magnification
scale_factor = 4.0

for idx, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < min_area:
        continue

    points = cnt.reshape(-1, 2)

    # Shift points so bottom-left is (0,0), then scale
    points_adjusted = np.empty_like(points, dtype=np.float32)
    points_adjusted[:, 0] = (points[:, 0] - min_x) * scale_factor
    points_adjusted[:, 1] = (points[:, 1] - min_y) * scale_factor

    # Flip Y to match DXF coordinate system (bottom-left origin)
    points_scaled = [(px, (5001 - min_y) * scale_factor - py) for px, py in points_adjusted]

    # Check if outer or hole
    if hierarchy[0][idx][3] == -1:
        msp.add_lwpolyline(points_scaled, close=True)
        outer_count += 1
    else:
        msp.add_lwpolyline(points_scaled, close=True)
        hole_count += 1

print(f"Exported {outer_count} outer region(s) and {hole_count} holes to DXF.")

# --- STEP 7: Save DXF ---
doc.saveas(output_dxf)
print(f"DXF saved as: {output_dxf}")
