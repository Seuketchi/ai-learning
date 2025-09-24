import cv2
import numpy as np
import json
import os

# Directories
input_dir = "dataset"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

output_file = "manual_params.json"

# Fixed parameters
dp = 1
param1 = 50
minRadius = 10
maxRadius = 40

# Max display size for tuning
MAX_WIDTH = 800
MAX_HEIGHT = 600

# Load existing manual parameters
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        manual_params = json.load(f)
else:
    manual_params = {}

# --- Helper functions ---
def detect_and_draw(image, param2, minDist):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 11)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )
    output = image.copy()
    count = 0
    filtered = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            if minRadius <= r <= maxRadius:
                filtered.append((x, y, r))
        count = len(filtered)
        for (x, y, r) in filtered:
            cv2.circle(output, (x, y), r, (0, 255, 0), 3)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
    return output, count, filtered, blurred

def resize_for_display(image):
    h, w = image.shape[:2]
    scale = min(MAX_WIDTH / w, MAX_HEIGHT / h, 1)
    if scale < 1:
        return cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

def save_panel(image, blurred, detected_image, mask, count, filename):
    panel1 = cv2.resize(image, (400, 400))
    panel2 = cv2.resize(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR), (400, 400))
    panel3 = cv2.resize(detected_image, (400, 400))
    panel4 = cv2.resize(mask, (400, 400))
    top_row = np.hstack([panel1, panel2])
    bottom_row = np.hstack([panel3, panel4])
    comparison = np.vstack([top_row, bottom_row])
    cv2.putText(comparison, f"Coins={count}", (10, 790),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    save_name = f"{filename}_detected.jpg"
    cv2.imwrite(os.path.join(output_dir, save_name), comparison)

# --- Loop through images ---
for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(input_dir, filename)
    image = cv2.imread(image_path)
    if image is None:
        continue

    # Load previous or default parameters
    params = manual_params.get(filename, {"param2": 17, "minDist": 50})
    param2 = params["param2"]
    minDist = params["minDist"]

    while True:
        detected_image, count, filtered, blurred = detect_and_draw(image, param2, minDist)

        # Create a temporary mask for visualization
        clean_mask = np.zeros_like(image)
        for (x, y, r) in filtered:
            cv2.circle(clean_mask, (x, y), r, (0, 255, 0), -1)

        display_img = resize_for_display(detected_image.copy())
        cv2.putText(display_img, f"param2={param2}, minDist={minDist}, Coins={count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Manual Tuner", display_img)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):  # Skip without saving
            break
        elif key == ord('u'):  # Increase param2
            param2 += 1
        elif key == ord('j'):  # Decrease param2
            param2 = max(1, param2 - 1)
        elif key == ord('i'):  # Increase minDist
            minDist += 1
        elif key == ord('k'):  # Decrease minDist
            minDist = max(1, minDist - 1)
        elif key == ord('s'):  # Save parameters and export result
            manual_params[filename] = {"param2": param2, "minDist": minDist}
            with open(output_file, "w") as f:
                json.dump(manual_params, f, indent=4)

            save_panel(image, blurred, detected_image, clean_mask, count, filename)
            print(f"[SAVED] {filename}: param2={param2}, minDist={minDist}, exported to results/")
            break

cv2.destroyAllWindows()
