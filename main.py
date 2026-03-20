from utils import *
import matplotlib.pyplot as plt
import cv2

image_paths = [
    'images/img1.jpg',
    'images/img2.jpg',
    'images/img3.jpg'
]

for i, path in enumerate(image_paths):

    gray = load_and_convert(path)

    eq = apply_equalization(gray)
    clahe_img = apply_clahe(gray)
    edges = detect_edges(gray)
    median = median_filter(gray)

    titles = ['Original',
              'Equalized',
              'CLAHE',
              'Edges',
              'Median Filter']
    images = [gray,
              eq,
              clahe_img,
              edges,
              median]

    plt.figure(figsize=(12,6))

    for j in range(5):
        plt.subplot(2,3,j+1)
        plt.imshow(images[j], cmap='gray')
        plt.title(titles[j])
        plt.axis('off')

    plt.show()

    cv2.imwrite(f'outputs/img{i+1}_median.png', median)