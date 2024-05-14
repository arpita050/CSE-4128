import cv2
import numpy as np
from matplotlib import pyplot as plt

# Butterworth notch filter
def notch_butterworth(uk, vk, M, N, D0=5.0, n=2.0):
    filter = np.ones((M, N))
    for u in range(M):
        for v in range(N):
            Dk = np.sqrt((u - M//2 - uk)**2 + (v - N//2 - vk)**2)
            Dkk = np.sqrt((u - M//2 + uk)**2 + (v - N//2 + vk)**2)
            filter[u, v] *= (1 / (1 + (D0 / Dk)**(2*n))) * (1 / (1 + (D0 / Dkk)**(2*n)))
    return filter

image_path = 'C:/Users/ASUS/Downloads/pnois1.jpg'
img = cv2.imread(image_path, 0)
if img is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

log = cv2.GaussianBlur(img, (3, 3), 0)
log = cv2.Laplacian(log, cv2.CV_64F)
log = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Fourier transform
M, N = log.shape
ft = np.fft.fft2(log)
ft_shift = np.fft.fftshift(ft)
magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(magnitude_spectrum_ac + 1)
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
ang = np.angle(ft_shift)
ang_ = cv2.normalize(ang, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Take user input
number_of_notches = int(input("Enter the number of notches: "))
notch_positions = []
for i in range(number_of_notches):
    uk = int(input(f"Enter uk for notch {i+1}: "))
    vk = int(input(f"Enter vk for notch {i+1}: "))
    notch_positions.append((uk, vk))

filter_img = np.ones((M, N))

for uk, vk in notch_positions:
    notch_filter = notch_butterworth(uk, vk, M, N, D0=5.0, n=2)
    filter_img *= notch_filter

filtered_ft_shift = filter_img * ft_shift

# Inverse Fourier transform
final_result = np.fft.ifftshift(filtered_ft_shift)
img_back = np.real(np.fft.ifft2(final_result))
img_back_scaled = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow("Input", img)
cv2.imshow("LoG Filtered", log)
cv2.imshow("Magnitude Spectrum", magnitude_spectrum)
cv2.imshow("Phase", ang_)
cv2.imshow("Filter", cv2.normalize(filter_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F))
cv2.imshow("Filtered Spectrum", cv2.normalize(np.abs(filtered_ft_shift), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F))
cv2.imshow("Inverse Transform", img_back_scaled)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
plt.figure(figsize=(20, 10))
plt.subplot(2, 4, 1), plt.imshow(img, cmap='gray'), plt.title('Input Image')
plt.subplot(2, 4, 2), plt.imshow(log, cmap='gray'), plt.title('LoG Filtered')
plt.subplot(2, 4, 3), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Magnitude Spectrum')
plt.subplot(2, 4, 4), plt.imshow(ang_, cmap='gray'), plt.title('Phase')
plt.subplot(2, 4, 5), plt.imshow(cv2.normalize(filter_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F), cmap='gray'), plt.title('Filter')
plt.subplot(2, 4, 6), plt.imshow(cv2.normalize(np.abs(filtered_ft_shift), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F), cmap='gray'), plt.title('Filtered Spectrum')
plt.subplot(2, 4, 7), plt.imshow(img_back_scaled, cmap='gray'), plt.title('Inverse Transform')
plt.show()

'''