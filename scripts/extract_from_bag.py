
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores
import os
import os.path as osp
import numpy as np
import cv2

# =============================================================================
# configuration
# =============================================================================

BAG_NAME = "bag_2025_6_4_15_27_55"
BAG_FOLDER = "../../../bags"
BAG_PATH = osp.join(BAG_FOLDER, BAG_NAME)
OUTPUT_DIR = osp.join("../data", BAG_NAME)

IMAGE_TOPIC = "/front_camera/rgb/image_raw"
CAMERA_INFO_TOPIC = "/front_camera/rgb/camera_info"
SAMPLE_RATE = 30

typestore = get_typestore(Stores.ROS2_HUMBLE)

with Reader(BAG_PATH) as reader:
    print("Topics:")
    for conn in reader.connections:
        print(f"- {conn.topic} ({conn.msgtype})")

# =============================================================================
# create output directories
# =============================================================================
os.makedirs(osp.join(OUTPUT_DIR, "raw_data", "images"), exist_ok=True)

print(f"Output directory: {osp.abspath(OUTPUT_DIR)}")


# =============================================================================
# camera intrinsics
# =============================================================================
camera_info = None

with Reader(BAG_PATH) as reader:
    for conn, timestamp, raw_data in reader.messages():
        if conn.topic == CAMERA_INFO_TOPIC:
            msg = typestore.deserialize_cdr(raw_data, conn.msgtype)
            camera_info = {
                'width': msg.width,
                'height': msg.height,
                'fx': msg.k[0],   # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
            }
            break

print(f"\nCamera: {camera_info['width']}x{camera_info['height']}")
print(f"Focal length: fx={camera_info['fx']:.2f}, fy={camera_info['fy']:.2f}")
print(f"Principal point: cx={camera_info['cx']:.2f}, cy={camera_info['cy']:.2f}")


# =============================================================================
# extract images
# =============================================================================
image_list = []
frame_count = 0
saved_count = 0

print(f"\nExtracting images (every {SAMPLE_RATE} frames)...")

with Reader(BAG_PATH) as reader:
    for conn, timestamp, raw_data in reader.messages():
        if conn.topic == IMAGE_TOPIC:
            if frame_count % SAMPLE_RATE == 0:
                msg = typestore.deserialize_cdr(raw_data, conn.msgtype)

                # Convert ROS image to numpy
                if msg.encoding == 'rgb8':
                    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif msg.encoding == 'bgr8':
                    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                elif msg.encoding == 'bgra8':
                    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                else:
                    print(f"Warning: Unknown encoding {msg.encoding}")
                    frame_count += 1
                    continue

                # Save image
                filename = f"frame_{saved_count:06d}.jpg"
                filepath = osp.join(OUTPUT_DIR, "raw_data", "images", filename)
                cv2.imwrite(filepath, image)

                # Store metadata
                image_list.append({
                    'image_path': osp.abspath(filepath),
                    'timestamp': timestamp,
                })

                saved_count += 1
                if saved_count % 50 == 0:
                    print(f"  Saved {saved_count} images...")

            frame_count += 1

print(f"\nExtracted {saved_count} images from {frame_count} frames.")

# =============================================================================
# write images.txt
# =============================================================================
images_txt = osp.join(OUTPUT_DIR, "images.txt")
with open(images_txt, "w") as f:
    f.write("timestamp, image_path\n")
    for img in image_list:
        f.write(f"{img['timestamp']}, {img['image_path']}\n")
print(f"Wrote: {osp.abspath(images_txt)}")