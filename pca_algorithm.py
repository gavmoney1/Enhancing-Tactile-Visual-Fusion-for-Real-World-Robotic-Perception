import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, color
from sklearn.decomposition import PCA

# PCA compression function
def apply_pca(image, components=50):
    """
    Compress a single grayscale image using PCA
    """
    pca = PCA(n_components=components)
    transformed = pca.fit_transform(image)
    pca_image = pca.inverse_transform(transformed)
    return pca_image

# Load the dataset
def load_images(input_root, output_root, target_size=(224,224)):
    images = []
    i=0

    # Walk through dataset
    for root, _, files in os.walk(input_root):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg",".png")):
                continue

            input_path = os.path.join(root, fname) # builds the full path to the input file
            relative_path = os.path.relpath(input_path, input_root) # finds its path relative to the input root
            output_path = os.path.join(output_root, relative_path) # creates the corresponding output path in another root folder 
            os.makedirs(os.path.dirname(output_path), exist_ok=True) #Ensures the output folder exists before saving a file there 

            # print(input_path)
            # return

            img = cv2.imread(input_path)
            if img is None:
                print(f"Could not open {input_path}")
                continue

            img_resized = transform.resize(img, target_size, anti_aliasing=True)
            img_gray = color.rgb2gray(img_resized)

            compressed_image = apply_pca(img_gray)
            images.append(compressed_image)

            cv2.imwrite(output_path, compressed_image)
            print(f"Saved compressed image: {output_path}")

            i += 1
            if i > 5: 
                return np.array(images)


    return np.array(images)

input_root = r"C:\Users\gabri\OneDrive\Documents\Fall25\Capstone\Development\masked_frames"
output_root = r"C:\Users\gabri\OneDrive\Documents\Fall25\Capstone\Development\pca_masked_frames"

# load_images(input_root, output_root)

# apply PCA to dataset 
compressed_images = load_images(input_root, output_root)
print("Compressed dataset shape:", compressed_images.shape)

# view images
plt.figure(figsize=(8,4))

plt.subplot(1,2,2)
plt.title("PCA Compressed (50 PCs)")
plt.imshow(compressed_images[1], cmap="gray")
plt.axis("off")

plt.show()
     
