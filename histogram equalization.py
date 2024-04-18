import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_pdf_cdf(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)
    return pdf, cdf

def histogram_equalization(image):
    pdf, cdf = calculate_pdf_cdf(image)
    equalized_image = np.interp(image.flatten(), range(256), 255*cdf).astype(np.uint8)
    return equalized_image.reshape(image.shape)

# Load the input color image
input_image = cv2.imread('color_img.jpg')
input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Separate RGB channels
r_channel = input_image[:,:,0]
g_channel = input_image[:,:,1]
b_channel = input_image[:,:,2]

# Equalize each channel separately
equalized_r = histogram_equalization(r_channel)
equalized_g = histogram_equalization(g_channel)
equalized_b = histogram_equalization(b_channel)

# Merge the equalized channels
equalized_image_rgb = cv2.merge([equalized_r, equalized_g, equalized_b])

# Convert the RGB image to HSV format
input_image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

# Separate channels
h_channel = input_image_hsv[:,:,0]
s_channel = input_image_hsv[:,:,1]
v_channel = input_image_hsv[:,:,2]

# Equalize the Value (V) channel
equalized_v = histogram_equalization(v_channel)

# Merge the equalized Value channel with the original Hue and Saturation channels
equalized_image_hsv = cv2.merge([h_channel, s_channel, equalized_v])
equalized_image_hsv_rgb = cv2.cvtColor(equalized_image_hsv, cv2.COLOR_HSV2RGB)

# Plot the images and histograms
plt.figure(figsize=(18, 10))

# Plot for RGB equalization
plt.subplot(2, 4, 1)
plt.imshow(input_image_rgb)
plt.title('Input Image')
plt.axis('off')

channels = ['R', 'G', 'B']
for i, channel in enumerate([equalized_r, equalized_g, equalized_b]):
    plt.subplot(2, 4, i + 2)
    plt.imshow(channel, cmap='gray')
    plt.title(f'Equalized {channels[i]} Channel')
    plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(equalized_image_rgb)
plt.title('Merged Equalized Image (RGB)')
plt.axis('off')

# Plot for HSV equalization
plt.subplot(2, 4, 6)
plt.imshow(input_image_rgb)
plt.title('Input Image (HSV)')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(equalized_v, cmap='gray')
plt.title('Equalized Value Channel (HSV)')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(equalized_image_hsv_rgb)
plt.title('Merged Equalized Image (HSV)')
plt.axis('off')

# Show histograms
plt.figure(figsize=(18, 6))

# Histogram of input image
plt.subplot(2, 2, 1)
plt.hist(input_image_rgb.ravel(), 256, [0, 256], color='blue', alpha=0.7)
plt.title('Histogram of Input Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Histogram of equalized image
plt.subplot(2, 2, 2)
plt.hist(equalized_image_rgb.ravel(), 256, [0, 256], color='green', alpha=0.7)
plt.title('Histogram of Equalized Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Histogram of CDF and PDF calculated from the input image
input_pdf, input_cdf = calculate_pdf_cdf(input_image_rgb)
plt.subplot(2, 2, 3)
plt.plot(input_pdf, color='blue', label='PDF')
plt.plot(input_cdf, color='red', label='CDF')
plt.title('PDF and CDF of Input Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability')
plt.legend()

# Histogram of CDF and PDF calculated from the equalized image
equalized_pdf, equalized_cdf = calculate_pdf_cdf(equalized_image_rgb)
plt.subplot(2, 2, 4)
plt.plot(equalized_pdf, color='green', label='PDF')
plt.plot(equalized_cdf, color='purple', label='CDF')
plt.title('PDF and CDF of Equalized Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability')
plt.legend()

plt.tight_layout()
plt.show()
