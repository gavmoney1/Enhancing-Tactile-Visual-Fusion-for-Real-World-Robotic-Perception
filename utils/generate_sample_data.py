from PIL import Image, ImageDraw
import random
import os

# Parameters
num_images = 999
classes = [0, 1, 2]
images_per_class = num_images // len(classes)  # 333 images per class
image_size = (224, 224)
output_dir = "shapes"
os.makedirs(output_dir, exist_ok=True)
label_file_path = os.path.join(output_dir, "labels.txt")

# Shape mapping per class
shapes = {
    0: "circle",
    1: "square",
    2: "triangle"
}

def draw_shape(draw, shape_type, size):
    if shape_type == "circle":
        bbox = [size*0.2, size*0.2, size*0.8, size*0.8]
        draw.ellipse(bbox, fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    elif shape_type == "square":
        bbox = [size*0.2, size*0.2, size*0.8, size*0.8]
        draw.rectangle(bbox, fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    elif shape_type == "triangle":
        points = [(size*0.5, size*0.2), (size*0.2, size*0.8), (size*0.8, size*0.8)]
        draw.polygon(points, fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))

# Generate images and labels
labels = []
img_count = 1
for cls in classes:
    for _ in range(images_per_class):
        img = Image.new("RGB", image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw_shape(draw, shapes[cls], image_size[0])
        img_filename = f"image_{img_count:04d}.jpg"
        img.save(os.path.join(output_dir, img_filename))
        labels.append(f"{img_filename},{cls}")
        img_count += 1

# Write labels.txt
with open(label_file_path, "w") as f:
    f.write("image,label\n")
    f.write("\n".join(labels))

print(f"Generated {num_images} images in '{output_dir}' with labels.txt")
