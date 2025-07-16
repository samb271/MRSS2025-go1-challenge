import cv2
import numpy as np
import os

from pathlib import Path


def generate_aruco_tags(output_dir="textures/aruco_tags", tag_size=512):
    """Generate ArUco tag images"""
    os.makedirs(output_dir, exist_ok=True)

    # Create ArUco dictionary (4x4 with 50 tags)
    aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_4X4_50)

    # Generate tags 0-5
    for tag_id in range(6):
        # Generate the tag
        tag_image = cv2.aruco.drawMarker(aruco_dict, tag_id, tag_size)

        # Add white border around the tag
        border_size = tag_size // 8
        bordered_image = np.ones((tag_size + 2 * border_size, tag_size + 2 * border_size), dtype=np.uint8) * 255
        bordered_image[border_size : border_size + tag_size, border_size : border_size + tag_size] = tag_image

        # Save as PNG
        filename = os.path.join(output_dir, f"aruco_tag_{tag_id}.png")
        cv2.imwrite(filename, bordered_image)
        print(f"Generated: {filename}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "textures" / "aruco_tags"

    generate_aruco_tags(output_dir=output_dir)
