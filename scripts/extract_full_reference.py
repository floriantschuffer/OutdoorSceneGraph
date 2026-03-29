#!/usr/bin/env python3
"""
Full reference script to extract data from ROS bag for OutdoorSceneGraph pipeline.
Copy sections step-by-step to extract_from_bag.py
"""

# =============================================================================
# STEP 1: Imports
# =============================================================================
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import numpy as np
import cv2
import os

# =============================================================================
# STEP 2: Configuration
# =============================================================================
BAG_PATH = "/Users/floriantschuffer/Downloads/bags/bag_2025_6_4_15_27_55"
OUTPUT_DIR = "/Users/floriantschuffer/huggingFaceVenv/src/OutdoorSceneGraph/Lamar/WARTHOG"
IMAGE_TOPIC = "/front_camera/rgb/image_raw"
CAMERA_INFO_TOPIC = "/front_camera/rgb/camera_info"
SAMPLE_RATE = 30  # Extract every N-th frame (30 = ~1fps from 30fps video)

# =============================================================================
# STEP 3: Create output directories
# =============================================================================
os.makedirs(os.path.join(OUTPUT_DIR, "raw_data", "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "proc", "meshes"), exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# =============================================================================
# STEP 4: Extract camera intrinsics
# =============================================================================
camera_info = None

with Reader(BAG_PATH) as reader:
    for conn, timestamp, rawdata in reader.messages():
        if conn.topic == CAMERA_INFO_TOPIC:
            msg = deserialize_cdr(rawdata, conn.msgtype)
            camera_info = {
                'width': msg.width,
                'height': msg.height,
                'fx': msg.k[0],   # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
            }
            break

print(f"Camera: {camera_info['width']}x{camera_info['height']}")
print(f"Focal length: fx={camera_info['fx']:.2f}, fy={camera_info['fy']:.2f}")
print(f"Principal point: cx={camera_info['cx']:.2f}, cy={camera_info['cy']:.2f}")

# =============================================================================
# STEP 5: Extract images
# =============================================================================
images_list = []
frame_count = 0
saved_count = 0

print(f"\nExtracting images (every {SAMPLE_RATE} frames)...")

with Reader(BAG_PATH) as reader:
    for conn, timestamp, rawdata in reader.messages():
        if conn.topic == IMAGE_TOPIC:
            if frame_count % SAMPLE_RATE == 0:
                msg = deserialize_cdr(rawdata, conn.msgtype)
                
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
                filepath = os.path.join(OUTPUT_DIR, "raw_data", "images", filename)
                cv2.imwrite(filepath, image)
                
                # Store metadata
                images_list.append({
                    'image_path': f"/images/{filename}",
                    'timestamp': timestamp
                })
                
                saved_count += 1
                if saved_count % 50 == 0:
                    print(f"  Saved {saved_count} images...")
            
            frame_count += 1

print(f"Extracted {saved_count} images from {frame_count} total frames")

# =============================================================================
# STEP 6: Write images.txt
# =============================================================================
images_txt = os.path.join(OUTPUT_DIR, "images.txt")
with open(images_txt, 'w') as f:
    f.write("timestamp, image_path\n")
    for img in images_list:
        f.write(f"{img['timestamp']}, {img['image_path']}\n")
print(f"Wrote: {images_txt}")

# =============================================================================
# STEP 7: Write sensors.txt
# =============================================================================
sensors_txt = os.path.join(OUTPUT_DIR, "sensors.txt")
with open(sensors_txt, 'w') as f:
    f.write("# sensor_id, name, model, param, width, height, fx, fy, cx, cy\n")
    f.write(f"0, front_camera, PINHOLE, 0, {camera_info['width']}, {camera_info['height']}, "
            f"{camera_info['fx']}, {camera_info['fy']}, {camera_info['cx']}, {camera_info['cy']}\n")
print(f"Wrote: {sensors_txt}")

# =============================================================================
# STEP 8: Write placeholder trajectories.txt (NEEDS SLAM!)
# =============================================================================
trajectories_txt = os.path.join(OUTPUT_DIR, "trajectories.txt")
with open(trajectories_txt, 'w') as f:
    f.write("timestamp, qw, qx, qy, qz, tx, ty, tz\n")
    for img in images_list:
        # Identity pose placeholder - REPLACE WITH SLAM OUTPUT!
        f.write(f"{img['timestamp']}, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n")
print(f"Wrote: {trajectories_txt} (PLACEHOLDER - needs SLAM poses!)")

# =============================================================================
# DONE - Summary
# =============================================================================
print("\n" + "="*60)
print("EXTRACTION COMPLETE")
print("="*60)
print(f"Images:      {saved_count}")
print(f"Output:      {OUTPUT_DIR}")
print(f"\nFiles created:")
print(f"  - images.txt")
print(f"  - sensors.txt")
print(f"  - trajectories.txt (placeholder)")
print(f"  - raw_data/images/*.jpg")
print(f"\nNEXT STEPS:")
print(f"  1. Run SLAM on LiDAR to get real camera poses")
print(f"  2. Update trajectories.txt with SLAM output")
print(f"  3. Generate mesh from point cloud")
print(f"  4. Place mesh in proc/meshes/mesh_simplified.ply")
