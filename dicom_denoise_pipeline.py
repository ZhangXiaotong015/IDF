import argparse
import torch
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
import numpy as np
import pydicom
from idf.utils.common import instantiate_from_config, load_state_dict
from idf.utils.misc import const_like
import cv2

def convert_denoised_to_dicom(denoised_png_paths, original_dicom_path, output_dicom_path):
    try:
        original_dicom = pydicom.dcmread(original_dicom_path, force=True)
        original_pixel_array = original_dicom.pixel_array
        num_frames = original_pixel_array.shape[0]

        frame_arrays = []
        for i, png_path in enumerate(sorted(denoised_png_paths)):
            img = Image.open(png_path).convert('L')
            img_array = np.array(img)
            frame_min = original_pixel_array[i].min()
            frame_max = original_pixel_array[i].max()
            mapped_array = frame_min + (img_array.astype(np.float32) / 255.0) * (frame_max - frame_min)
            frame_arrays.append(mapped_array.astype(original_pixel_array.dtype))

        combined_array = np.stack(frame_arrays, axis=0)
        original_dicom.PixelData = combined_array.tobytes()
        original_dicom.Rows, original_dicom.Columns = combined_array.shape[1:3]
        original_dicom.NumberOfFrames = num_frames
        original_dicom.SeriesDescription = "Denoised Image"
        original_dicom.save_as(output_dicom_path)
        print(f"Saved {output_dicom_path}")
        return True
    except Exception as e:
        print(f"Failed {original_dicom_path}: {e}")
        return False

def process_dicom_directory(input_dir, output_dir, model, device, config):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dicom_files = []
    for ext in ['.dcm', '.dicom']:
        dicom_files.extend(list(input_path.rglob(f"*{ext}")))

    if not dicom_files:
        print("No DICOM files found with .dcm or .dicom extension. Trying to detect DICOM files by content...")
        all_files = list(input_path.rglob("*"))
        for file_path in all_files:
            if file_path.is_file() and file_path.stat().st_size > 100:
                try:
                    with open(file_path, 'rb') as f:
                        preamble = f.read(132)
                        if len(preamble) >= 132 and preamble[128:132] == b'DICM':
                            dicom_files.append(file_path)
                except:
                    continue

    if not dicom_files:
        print("No DICOM files found!")
        return False

    for dicom_file in dicom_files:
        print(f"Processing {dicom_file}")
        dicom_data = pydicom.dcmread(dicom_file, force=True)
        pixel_array = dicom_data.pixel_array
        num_frames = pixel_array.shape[0]

        temp_pngs = []
        for i in range(num_frames):
            png_path = output_path / f"{dicom_file.stem}_frame{i}.png"
            Image.fromarray(pixel_array[i]).save(png_path)
            temp_pngs.append(str(png_path))

        denoised_paths = []
        for png_path in temp_pngs:
            img = Image.open(png_path).convert("RGB")
            tensor = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0).float().to(device)/255.0
            output = model(tensor, adaptive_iter=False, max_iter=3)
            if isinstance(output, tuple):
                output = output[0]
            out_img = output.squeeze(0).detach().cpu().clamp(0,1).numpy().transpose(1,2,0)*255
            denoised_path = output_path / f"{Path(png_path).stem}_denoised.png"
            Image.fromarray(out_img.astype(np.uint8)).save(denoised_path)
            denoised_paths.append(str(denoised_path))

        convert_denoised_to_dicom(denoised_paths, dicom_file, output_path / f"{dicom_file.stem}_denoised.dcm")

    return True


def process_dicom_directory_V2(input_dir, output_dir, model, device, config):
    import torch
    import numpy as np

    def normalize(x, reverse=False):
        # 保证输入是 torch.Tensor
        if isinstance(x, tuple):
            # 如果是 tuple，取第一个元素
            x = x[0]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        # data normalization
        sigma_data = 0.5
        mu_data = 0.0
        raw_std = [0.19113504, 0.19113504, 0.19113504]
        raw_mean = [0.55234206, 0.55234206, 0.55234206]
        data_scale = np.float32(sigma_data) / np.float32(raw_std)
        data_bias = np.float32(mu_data) - np.float32(raw_mean) * data_scale

        if not reverse:
            if data_scale is not None:
                x = x * torch.tensor(data_scale, dtype=x.dtype, device=x.device).reshape(1, -1, 1, 1)
            if data_bias is not None:
                x = x + torch.tensor(data_bias, dtype=x.dtype, device=x.device).reshape(1, -1, 1, 1)
        else:
            if data_bias is not None:
                x = x - torch.tensor(data_bias, dtype=x.dtype, device=x.device).reshape(1, -1, 1, 1)
            if data_scale is not None:
                x = x / torch.tensor(data_scale, dtype=x.dtype, device=x.device).reshape(1, -1, 1, 1)

        return x

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dicom_files = []
    for ext in ['.dcm', '.dicom']:
        dicom_files.extend(list(input_path.rglob(f"*{ext}")))

    if not dicom_files:
        print("No DICOM files found with .dcm or .dicom extension. Trying to detect DICOM files by content...")
        all_files = list(input_path.rglob("*"))
        for file_path in all_files:
            if file_path.is_file() and file_path.stat().st_size > 100:
                try:
                    with open(file_path, 'rb') as f:
                        preamble = f.read(132)
                        if len(preamble) >= 132 and preamble[128:132] == b'DICM':
                            dicom_files.append(file_path)
                except:
                    continue

    if not dicom_files:
        print("No DICOM files found!")
        return False

    for dicom_file in dicom_files:
        print(f"Processing {dicom_file}")
        dicom_data = pydicom.dcmread(dicom_file, force=True)
        pixel_array = dicom_data.pixel_array
        num_frames = pixel_array.shape[0]

        temp_pngs = []
        for i in range(num_frames):
            png_path = output_path / f"{dicom_file.stem}_frame{i}.png"
            Image.fromarray(pixel_array[i]).save(png_path)
            temp_pngs.append(str(png_path))

        denoised_paths = []
        for png_path in temp_pngs:
            img = Image.open(png_path).convert("RGB")
            tensor = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0).float().to(device)/255.0
            pred = model(tensor, adaptive_iter=False, max_iter=3)

            pred = normalize(pred, reverse=True)

            # -------------------------
            # Post-processing Enhancement V1 (CLAHE + Unsharp Mask)
            # -------------------------
            enhanced_list = []
            for i in range(pred.size(0)):  # 遍历 batch
                img = pred[i].detach().cpu().numpy()  # (3, H, W)
                img = np.transpose(img, (1, 2, 0))  # 转成 (H, W, 3)
                img = (img * 255).astype(np.uint8)

                # 取灰度通道
                pred_gray = img[:, :, 0]

                # CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_gray = clahe.apply(pred_gray)

                # 转回RGB三通道
                enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)

                # Unsharp Mask
                # blurred = cv2.GaussianBlur(enhanced_rgb, (5, 5), 1.0)
                blurred = cv2.GaussianBlur(cv2.copyMakeBorder(enhanced_rgb, 5, 5, 5, 5, cv2.BORDER_REFLECT), (5, 5),
                                           1.0)
                blurred = blurred[5:-5, 5:-5]  # 去掉填充区域
                alpha = 1.5
                sharpened = cv2.addWeighted(enhanced_rgb, 1 + alpha, blurred, -alpha, 0)

                # 转回 torch.Tensor (C, H, W)
                sharpened_tensor = torch.from_numpy(sharpened).float().permute(2, 0, 1) / 255.0
                enhanced_list.append(sharpened_tensor)

            # 拼回 batch
            enhanced_pred = torch.stack(enhanced_list, dim=0)

            output = torch.clamp(enhanced_pred, 0.0, 1.0)  # (1,3,H,W)

            if isinstance(output, tuple):
                output = output[0]
            out_img = output.squeeze(0).detach().cpu().clamp(0,1).numpy().transpose(1,2,0)*255
            denoised_path = output_path / f"{Path(png_path).stem}_denoised.png"
            Image.fromarray(out_img.astype(np.uint8)).save(denoised_path)
            denoised_paths.append(str(denoised_path))

        convert_denoised_to_dicom(denoised_paths, dicom_file, output_path / f"{dicom_file.stem}_denoised.dcm")

    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/models/idfnet.yaml")
    parser.add_argument("--checkpoint", type=str, default="pretrained_models/epoch=561-step=50000.ckpt")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_state_dict(model, state_dict, strict=True)
    model.freeze().eval()
    device = torch.device(args.device if torch.cuda.is_available() and args.device=="cuda" else "cpu")
    model.to(device)

    # process_dicom_directory(args.input_dir, args.output_dir, model, device, config)
    process_dicom_directory_V2(args.input_dir, args.output_dir, model, device, config)

if __name__ == "__main__":
    main()
