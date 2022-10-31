import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
# print(digits)
# print(digits.DESCR)
# print(digits.data)
# print(digits.target)

# -----------------------------------------------

plt.gray()

plt.matshow(digits.images[100])

# plt.show()

# -----------------------------------------------

# Take a look at 64 sample images.
# Figure size (width, height)

fig = plt.figure(figsize=(6, 6))

# Adjust the subplots

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images

for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position

    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])

    # Display an image at the i-th position

    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # Label the image with the target value

    ax.text(0, 7, str(digits.target[i]))

# plt.show()

# -----------------------------------------------

model = KMeans(n_clusters=10, random_state=42)

model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))

fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)

    # Display images
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

# plt.show()

# -----------------------------------------------

new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,2.49,6.09,4.94,2.06,0.53,0.00,0.00,0.60,6.93,4.10,4.87,5.71,7.23,2.88,0.00,1.52,6.16,0.00,0.00,0.00,2.20,6.02,0.00,1.52,6.09,0.00,0.00,0.00,3.04,4.72,0.00,0.76,7.00,1.89,0.00,0.45,6.24,2.43,0.00,0.00,2.18,7.00,4.87,6.77,4.39,0.00,0.00,0.00,0.00,1.28,3.04,1.29,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.30,2.96,3.04,2.49,0.00,0.00,0.00,0.00,0.61,4.49,4.57,7.39,0.68,0.00,0.00,0.00,0.00,0.00,3.39,6.61,0.15,0.00,0.00,0.00,0.00,4.40,7.62,6.77,5.54,0.45,0.00,0.00,0.00,1.44,2.28,2.26,7.30,1.06,0.00,0.00,0.08,0.91,2.74,6.54,4.07,0.00,0.00,0.00,3.80,7.01,5.33,2.27,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,3.41,6.10,5.94,4.57,2.28,0.00,0.00,0.00,7.23,1.82,1.75,4.03,6.39,0.00,0.00,0.00,6.69,4.79,3.04,5.93,6.86,0.00,0.00,0.00,0.52,3.65,4.57,3.87,6.86,0.00,0.00,0.00,0.00,0.00,0.00,0.08,7.61,0.00,0.00,0.00,0.00,0.00,0.00,0.00,7.61,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.52,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.15,3.65,3.73,2.05,0.00,0.00,0.00,0.00,0.15,3.65,3.88,6.77,3.10,0.00,0.00,0.00,0.00,0.00,0.00,2.42,6.09,0.00,0.00,0.00,0.00,0.00,1.13,6.47,3.40,0.00,0.00,0.00,1.14,5.01,7.47,6.08,3.81,3.80,1.90,0.00,1.44,4.57,4.57,4.49,3.81,3.80,1.90,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00]
])

new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3,end='')




