# usage: python png2dicom.py -input input_directory -output output_directory
# python -m data_process.png2dicom -input="d:\code\C_ARM_denosing\idf\data\output_vedio" -output="D:\code\C_ARM_denosing\idf\data"

import os
import pydicom
import numpy as np
from pydicom.dataset import Dataset, FileDataset
from datetime import datetime
from PIL import Image
import argparse

def is_leaf_directory(path):
    return all(not os.path.isdir(os.path.join(path, f)) for f in os.listdir(path))

def load_png_as_array(path):
    img = Image.open(path).convert("L")
    return np.array(img)

def create_multiframe_dicom(frames, output_path):
    if len(frames) == 0:
        print(f"No frames to save for {output_path}")
        return

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Basic DICOM attributes
    ds.Modality = "OT"
    ds.ContentDate = datetime.now().strftime("%Y%m%d")
    ds.ContentTime = datetime.now().strftime("%H%M%S")

    num_frames = len(frames)
    height, width = frames[0].shape

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = height
    ds.Columns = width
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.NumberOfFrames = str(num_frames)

    pixel_data = np.stack(frames, axis=0).astype(np.uint8)
    ds.PixelData = pixel_data.tobytes()

    ds.ensure_file_meta()

    ds.save_as(output_path, write_like_original=False)

    print(f"Saved DICOM: {output_path} ({num_frames} frames)")


def process_directory(input_root, output_root):
    for dirpath, dirnames, filenames in os.walk(input_root):
        if not is_leaf_directory(dirpath):
            continue

        # 找到所有 PNG
        png_files = sorted(
            [f for f in filenames if f.lower().endswith(".jpg")],
            key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)  # 按数字排序
        )

        if not png_files:
            continue

        # 读取所有帧
        frames = []
        for f in png_files:
            img_path = os.path.join(dirpath, f)
            frames.append(load_png_as_array(img_path))

        # 输出路径
        relative = os.path.relpath(dirpath, input_root)
        output_dir = os.path.join(output_root, relative)
        os.makedirs(output_dir, exist_ok=True)

        dicom_name = os.path.basename(dirpath) + "_T1_Round2.dcm"
        output_path = os.path.join(output_dir, dicom_name)

        create_multiframe_dicom(frames, output_path)

# if __name__ == "__main__":
#     input_root = r"D:\code\C_ARM_denosing\idf\data\output_vedio"
#     output_root = r"D:\code\C_ARM_denosing\idf\data"
#     process_directory(input_root, output_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PNG images to multiframe DICOM files')
    parser.add_argument('-input', '--input', required=True, help='Input directory path')
    parser.add_argument('-output', '--output', required=True, help='Output directory path')
    
    args = parser.parse_args()
    
    input_root = args.input
    output_root = args.output
    
    process_directory(input_root, output_root)
