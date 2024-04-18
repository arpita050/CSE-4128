import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


def generate_hist(image):
    hist = np.zeros(256)
    for pixel_value in image.flatten():
        hist[pixel_value] += 1
    hist = hist / float(np.sum(hist))
    return hist

def find_closest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index

def hist_match(input_image, target_histogram):
    
    input_hist = generate_hist(input_image)
    input_cdf = np.cumsum(input_hist)
    
    target_cdf = np.cumsum(target_histogram)

    mapp = np.zeros(256, dtype=int)
    for i in range(256):
        mapp[i] = find_closest(target_cdf, input_cdf[i])

    output_img = np.array([mapp[int(value)] for value in input_image.flatten()])
    output = output_img.reshape(input_image.shape)
    output_hist = generate_hist(output)
    
    return output, input_hist, output_hist


img = cv.imread('input.jpg', cv.IMREAD_GRAYSCALE)

#Erlang
k = int(input("Enter shape parameter (k): "))
mu = float(input("Enter scale parameter (mu): "))

#Target histogram
target_hist = np.zeros(256)
for i in range(256):
    target_hist[i] = (i**(k-1) * np.exp(-i/mu)) / (math.factorial(k-1) * (mu**k))
target_hist /= np.sum(target_hist)

matched_img, input_hist, output_hist = hist_match(img, target_hist)

plt.figure(figsize=(15, 10))
plt.subplot(3, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Input Image")
plt.axis('off')

plt.subplot(3, 3, 2)
plt.plot(input_hist)
plt.title("Input Histogram")

plt.subplot(3, 3, 3)
plt.plot(np.cumsum(input_hist))
plt.title("Input CDF")

plt.subplot(3, 3, 4)
plt.imshow(matched_img, cmap='gray')
plt.title("Matched Output Image")
plt.axis('off')

plt.subplot(3, 3, 5)
plt.plot(output_hist)
plt.title("Output Histogram")

plt.subplot(3, 3, 6)
plt.plot(np.cumsum(output_hist))
plt.title("Output CDF")

plt.subplot(3, 3, 7)
plt.plot(target_hist)
plt.title("Target Histogram")

plt.subplot(3, 3, 8)
plt.plot(np.cumsum(target_hist))
plt.title("Target CDF")

plt.tight_layout()
plt.show()
