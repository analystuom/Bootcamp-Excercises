import numpy as np
from matplotlib import pyplot as plt

# Step 1: Load Data
file_path = "./full-numpy_bitmap-bus.npy"
images = np.load(file_path).astype(np.float32)
print(images.shape)
train_images = images[:-10]
test_images = images[-10:]

# Step 2 and 3
# Calculate mean pixels and reshape for visualising images
avg_image = np.mean(train_images, axis=0)
avg_image = avg_image.reshape(28, 28)
# plt.imshow(avg_image)
# plt.show()

# Step 4: Calculate cosine similarity of test image and average image:
index = 4
test_image = test_images[index]
score = np.dot(test_image.flatten(), avg_image.flatten())
print(score)

# Step 5 and 6:
categories = ["bus", "cake", "clock", "cookie", "cow", "dishwasher", "dog", "elephant",
              "flashlight", "guitar"]
score_dict = {}
avg_images = []
for category in categories:
    file_path = "./full-numpy_bitmap-{}.npy".format(category)
    images = np.load(file_path).astype(np.float32)
    avg_image = np.mean(images, axis=0)
    avg_images.append(avg_image.reshape(28, 28))
    test_image = images[1]
    score = np.dot(test_image.flatten(), avg_image.flatten())
    score_dict[category] = score
print(score_dict)

plt.figure(figsize=(10, 8))
for i, avg_img in enumerate(avg_images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(avg_img)
    plt.axis('off')
    plt.title(categories[i])
plt.tight_layout()
plt.show()
