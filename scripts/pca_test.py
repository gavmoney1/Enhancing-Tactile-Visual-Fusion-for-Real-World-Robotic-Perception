import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from sklearn.decomposition import PCA

image_path = r"Replace with image path"
image = io.imread(image_path)
# rgb to grayscale 
gray_image = rgb2gray(image)
height, width = gray_image.shape
print(gray_image.shape)
plt.imshow(gray_image, cmap="gray")
plt.axis("off")
plt.show()

pca = PCA(n_components=50)
transformed = pca.fit_transform(gray_image)

pca_image = pca.inverse_transform(transformed)

print(pca_image.shape)
print(pca.n_components)
plt.imshow(pca_image, cmap="gray")
plt.axis("off")
plt.show()


