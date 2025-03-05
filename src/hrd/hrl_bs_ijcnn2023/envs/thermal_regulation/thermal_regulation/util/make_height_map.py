
import png
import numpy as np

print("Generate New Height Map.")

max_height = 200

image = np.random.randint(0, max_height, (50, 50))

png.fromarray(image.tolist(), "L").save("hill_height.png")

print("hill_height.png saved. put this ino envs/models to configure height map.")
