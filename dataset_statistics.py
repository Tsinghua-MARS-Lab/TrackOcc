import os
import glob
from pathlib import Path

def dataset_statistics(data_root):
    """
    Dataset statistics
    
    Args:
        data_root: Root directory of the dataset
    
    Returns:
        dict: Dictionary containing statistics
    """
    # Ensure the path exists
    data_root = Path(data_root).expanduser()
    if not data_root.exists():
        print(f"Error: Path {data_root} does not exist")
        return None
    
    # Count the number of images in image_0 directory
    image_0_path = data_root / "image_0"
    if not image_0_path.exists():
        print(f"Error: image_0 directory does not exist at {image_0_path}")
        return None
    
    # Get all jpg files
    jpg_files = glob.glob(str(image_0_path / "*.jpg"))
    
    # 1. Total image count (image_0 file count * 5)
    image_0_count = len(jpg_files)
    total_images = image_0_count * 5
    
    # 2. Count number of scenes
    # Image naming format: abbbccc.jpg, extract abbb as scene ID
    scenes = set()
    for filename in jpg_files:
        basename = os.path.basename(filename)
        if len(basename) >= 7:  # Ensure sufficient length
            # Extract the first 4 digits as scene number
            scene_num = basename[:4]
            if scene_num.isdigit():
                scenes.add(scene_num)
    
    scene_count = len(scenes)
    
    # Print statistics
    print("=== Dataset Statistics ===")
    print(f"Dataset path: {data_root}")
    print(f"Total images: {total_images}")
    print(f"Number of scenes: {scene_count}")
    # print(f"Scene ID range: {min(scenes) if scenes else 'None'} - {max(scenes) if scenes else 'None'}")
    
    return {
        # 'image_0_count': image_0_count,
        'total_images': total_images,
        'scene_count': scene_count,
        'scenes': sorted(list(scenes))
    }

if __name__ == "__main__":
    # Dataset path
    data_root = "~/code/occ_tracking/TrackOcc/data/TrackOcc-waymo/kitti_format/training/"
    
    # Run statistics
    stats = dataset_statistics(data_root)
    
    # if stats:
    #     print("\n=== Detailed Information ===")
    #     print(f"Scene list: {stats['scenes']}")