import os
import cv2
import numpy as np

# high
# A = 3.50e-5
# B = 0.065
# C = -1.87e-6

# # mid A=1.75 cfd_add_noise_a_1_75
A = 1.75e-5
B = 0.065
C = -1.87e-6

# low A=0.9 C=1 cfd_add_noise_a_0_9
# A = 0.9e-5
# B = 0.065
# C = -1e-6

def add_poisson_gaussian_noise(img):
    # 1. Poisson
    scale = 255.0
    poisson_input = img * scale
    poisson_noisy = np.random.poisson(poisson_input).astype(np.float32) / scale

    # 2. Gaussian
    # σ² = A*(x + B) + C
    variance = A * (img + B) + C
    variance = np.clip(variance, 0, None)
    sigma = np.sqrt(variance)

    gaussian_noise = np.random.normal(0, sigma).astype(np.float32)

    noisy = poisson_noisy + gaussian_noise

    noisy = np.clip(noisy, 0, 1)

    return noisy


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            path = os.path.join(input_folder, filename)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0

            noisy_img = add_poisson_gaussian_noise(img)

            out_path = os.path.join(output_folder, filename)
            cv2.imwrite(out_path, (noisy_img * 255).astype(np.uint8))

            print(f"Processed: {filename}")


if __name__ == "__main__":
    input_folder = r"/data/ASD/train"
    output_folder = r"/data/asd_add_noise_a_1_75"
    print(f"Processing {input_folder}")
    print(f"Output folder {output_folder}")
    process_folder(input_folder, output_folder)
