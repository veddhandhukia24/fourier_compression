# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

def fourier_compress(image_path, keep_fraction):
    """
    Performs Fourier Transform-based image compression.

    Args:
        image_path (str): The path to the input image.
        keep_fraction (float): The fraction of low-frequency coefficients to keep (e.g., 0.1 for 10%).
    """
    try:
        # Step 1: Read the input image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not open or find the image at '{image_path}'")
            return
    except Exception as e:
        print(f"An error occurred while reading the image: {e}")
        return

    # --- COMPRESSION PROCESS ---

    # Step 2: Apply 2D Fast Fourier Transform (FFT)
    # This converts the image from the spatial domain to the frequency domain
    f_transform = np.fft.fft2(img)
    
    # Shift the zero-frequency component (DC component) to the center for easier filtering
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Step 3 & 4: Frequency Analysis and Compression
    # Create a mask to filter out high-frequency components
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2

    # Create a rectangular mask centered in the middle
    mask = np.zeros((rows, cols), np.uint8)
    
    # Define the area of the mask to keep (low frequencies)
    mask[int(crow - rows * keep_fraction / 2):int(crow + rows * keep_fraction / 2),
         int(ccol - cols * keep_fraction / 2):int(ccol + cols * keep_fraction / 2)] = 1

    # Apply the mask to the shifted Fourier Transform, effectively setting high frequencies to zero
    f_transform_shifted_masked = f_transform_shifted * mask

    # --- RECONSTRUCTION PROCESS ---

    # Step 5: Reconstruct the image
    # Inverse shift to move the DC component back to the top-left corner
    f_transform_unmasked = np.fft.ifftshift(f_transform_shifted_masked)
    
    # Apply Inverse FFT to convert from frequency domain back to spatial domain
    img_reconstructed = np.fft.ifft2(f_transform_unmasked)
    
    # Take the absolute value to handle complex numbers and convert to an 8-bit image format
    img_reconstructed = np.abs(img_reconstructed).astype(np.uint8)

    # Step 6: Display the results
    plt.figure(figsize=(10, 5)) # Create a figure to hold the plots
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.xticks([]), plt.yticks([]) # Hide axes ticks

    plt.subplot(1, 2, 2)
    plt.imshow(img_reconstructed, cmap='gray')
    plt.title(f'Compressed (Keeping {keep_fraction*100:.1f}% of Frequencies)')
    plt.xticks([]), plt.yticks([])

    # Add a title for the whole window
    plt.suptitle('Fourier Transform Image Compression')
    plt.show()


if __name__ == '__main__':
    # --- DEMONSTRATION SETUP ---
    # 1. Make sure you have an image file named 'lena.png' in the same folder as this script.
    #    You can download a standard 'lena.png' test image from the internet.
    # 2. Change the 'keep' value below to see how it affects compression quality.
    
    IMAGE_FILE = 'lena.png' # You can change this to your image file
    KEEP_PERCENT = 0.1     
    
    fourier_compress(IMAGE_FILE, KEEP_PERCENT)