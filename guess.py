from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import sys
import ascii_magic

imagePath = sys.argv[1]

# Load the model
model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open(imagePath)
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
# image.show()
ascii_magic.to_terminal(ascii_magic.from_image(image, columns=50))

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)

lines = [line.rstrip('\n') for line in open('labels.txt')]
lines = [line.split()[1] for line in lines]

print('\n', '*' * 100, '\n')

for index, item in enumerate(prediction[0]):
  print(lines[index].rjust(16), ("%.0f" % (item * 100) + "%").rjust(6))

