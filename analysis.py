import cv2
import numpy as np
from utils import *

image_paths = [
    'images/img1.jpg',
    'images/img2.jpg',
    'images/img3.jpg'
]

def analyze_image(gray):
    mean = np.mean(gray)
    std = np.std(gray)
    return mean, std

for i, path in enumerate(image_paths):

    gray = load_and_convert(path)

    eq = apply_equalization(gray)
    clahe_img = apply_clahe(gray)

    print(f"\nImage {i+1} Analysis:")

    m1, s1 = analyze_image(gray)
    m2, s2 = analyze_image(eq)
    m3, s3 = analyze_image(clahe_img)

    print(f"Original   → Mean: {m1:.2f}, Std: {s1:.2f}")
    print(f"Equalized  → Mean: {m2:.2f}, Std: {s2:.2f}")
    print(f"CLAHE      → Mean: {m3:.2f}, Std: {s3:.2f}")