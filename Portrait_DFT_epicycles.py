# -*- coding: utf-8 -*-
"""
Created on Wed May 15 00:50:17 2024

@author: the great Arpita
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 14 23:23:24 2024

@author: the great Arpita
"""

# Import necessary libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import tau
import matplotlib.animation as animation

# Step 1: Load and Display the Image
img = cv2.imread("face.jpeg")  # Read the image
cv2.imshow('Input Image', img)  # Show the image
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Input Image.jpeg", img)  # Save the image

# Step 2: Convert to Grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
cv2.imshow('Grayscaled Image', img_gray)  # Show the grayscale image
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Grayscaled Image.jpeg", img_gray)  # Save the grayscale image

# Step 3: Edge Detection
canny_edges = cv2.Canny(img_gray, 120, 220)  # Detect edges
cv2.imshow('Edge Detected Image', canny_edges)  # Show the edge-detected image
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Canny Edge Detected Image.jpeg", canny_edges)  # Save the edge-detected image

# Step 4: Find Edge Points
edge_pts = np.argwhere(canny_edges == 255)  # Get coordinates of edge points

# Function to interpolate points
def interpolation(time, time_list, real, imag):
    return np.interp(time, time_list, real + 1j * imag)

# Function to create a path connecting all edge points
def path_trace(pts):
    total_points = pts.shape[0]
    path = np.zeros((total_points, 2), dtype=np.int64)
    path[0] = pts[0]
    added_to_path = 1
    not_added_to_path = pts[1:].astype(np.int64)
    vector_lengths = np.zeros((total_points,), dtype=np.float64)

    while added_to_path != total_points:  # Continues until all points are added to the path
        r = 5
        empty_circle = True
        d = np.abs(path[added_to_path - 1] - not_added_to_path)  # Calculate distances
        vector_len = np.zeros((d.shape[0],), dtype=np.float64)
        while empty_circle:  # Radius Expansion Loop
            for i in range(d.shape[0]):
                if d[i, 0] <= r and d[i, 1] <= r:
                    empty_circle = False
                    vector_len[i] = np.sqrt(d[i, 0] ** 2 + d[i, 1] ** 2)
                else:
                    vector_len[i] = 10000  # Exclude these points from being selected
            r += 5

        d_min = np.argsort(vector_len)[0]  # The nearest point within the current radius
        path[added_to_path] = not_added_to_path[d_min]
        vector_lengths[added_to_path] = vector_len[d_min]
        not_added_to_path = np.delete(not_added_to_path, d_min, axis=0)
        added_to_path += 1

    return path

# Step 5: Create the Path
path = path_trace(edge_pts)  # Generate the path from edge points

# Step 6: Prepare Data for Plotting
x_list, y_list = path[:, 1], -path[:, 0]  # Invert y-coordinates to match the image coordinate system
x_list, y_list = x_list - np.mean(x_list), y_list - np.mean(y_list)  # Center the coordinates around the origin

# Step 7: Plot and Save Contour Image
figure = plt.figure(facecolor='black')  # Create a figure with black background
axes = figure.add_subplot(1, 1, 1, facecolor='black')  # Add subplot with black background
axes.plot(x_list, y_list, 'r-')  # Plot the path in red

xlim_data, ylim_data = plt.xlim(), plt.ylim()  # Get plot limits
plt.axis('off')  # Hide axes
plt.savefig("contoured_image.png", bbox_inches='tight', pad_inches=0, dpi=300)  # Save the plot
plt.show()

# Step 8: Time Data for Fourier Series
time_list = np.linspace(0, tau, len(x_list))  # Create a time list from 0 to 2*pi

# Function to compute Fourier coefficients using FFT
def compute_fourier_coefficients(x_list, y_list, order):
    z = x_list + 1j * y_list
    coeffs = np.fft.fft(z) / len(z)
    coeffs = np.fft.fftshift(coeffs)  # Shift zero frequency component to the center
    middle = len(coeffs) // 2
    return coeffs[middle - order: middle + order + 1]

# Step 9: Compute Fourier Coefficients
order = 100  # Number of Fourier coefficients
fourier_coeff = compute_fourier_coefficients(x_list, y_list, order)  # Calculate coefficients
print("Coefficients generation done")

# Step 10: Prepare for Animation
point_x, point_y = [], []  # Lists to store drawing points

# Create a new figure for animation
figure, axes = plt.subplots(facecolor='black')

# Create green circles and lines for epicycles
circles = [axes.plot([], [], 'g-')[0] for i in range(-order, order + 1)]
circle_lines = [axes.plot([], [], 'g-')[0] for i in range(-order, order + 1)]

# Create red lines for the final drawing
final_fig, = axes.plot([], [], 'r-', linewidth=0.5)
original_fig, = axes.plot([], [], 'r-', linewidth=0.5)

# Set plot limits and aspect ratio
axes.set_xlim(xlim_data[0] - 300, xlim_data[1] + 300)
axes.set_ylim(ylim_data[0] - 300, ylim_data[1] + 300)
axes.set_aspect('equal')
axes.set_facecolor('black')

# Step 11: Animation Setup
frames_num = 200  # Number of frames in the animation

# Function to sort coefficients
def c_sort(coeffs):
    new_c = []
    new_c.append(coeffs[order])
    for i in range(1, order + 1):
        new_c.extend([coeffs[order + i], coeffs[order - i]])
    return np.array(new_c)

# Function to create each frame of the animation
def frame_create(i, time, coeffs):
    t = time[i]  # Get the current time
    exp_term = np.array([np.exp(n * t * 1j) for n in range(-order, order + 1)])
    coeffs = c_sort(coeffs * exp_term)  # Rotate the circles

    x_coeffs, y_coeffs = np.real(coeffs), np.imag(coeffs)
    center_x, center_y = 0, 0  # Center points for the first circle

    # Draw all circles (epicycles)
    for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
        r = np.linalg.norm([x_coeff, y_coeff])  # Calculate radius
        theta = np.linspace(0, tau, num=50)  # Theta from 0 to 2*pi
        x, y = center_x + r * np.cos(theta), center_y + r * np.sin(theta)
        circles[i].set_data(x, y)  # Set circle data

        x, y = [center_x, center_x + x_coeff], [center_y, center_y + y_coeff]
        circle_lines[i].set_data(x, y)  # Set line data

        center_x, center_y = center_x + x_coeff, center_y + y_coeff  # Update center points

    # Store drawing points
    point_x.append(center_x)
    point_y.append(center_y)
    final_fig.set_data(point_x, point_y)  # Set final drawing data

# Step 12: Generate and Save Animation
time = np.linspace(0, tau, num=frames_num)  # Time values for the animation
anim = animation.FuncAnimation(figure, frame_create, frames=frames_num, fargs=(time, fourier_coeff), interval=100)

# Save the animation with high resolution
anim.save('face_high_res.mp4', dpi=300, extra_args=['-vcodec', 'libx264'])

print("Completed: face_high_res.mp4")
plt.show()
