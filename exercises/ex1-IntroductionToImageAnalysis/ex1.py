#%% imports
from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

#%% Ex. 1 - Reading images
# Directory containing data and images
in_dir = "data/"

# X-ray image
im_name = "metacarpals.png"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

# %% Ex. 2 - Shape

print(im_org.shape)
# %% Ex. 3 - Pixel data type
print(im_org.dtype)
# %% Ex. 4 - Show images (in grey scale)
io.imshow(im_org)
plt.title('Metacarpal image')
io.show()

# %% Ex. 5 - Color maps

io.imshow(im_org, cmap="jet")
plt.title('Metacarpal image (with colormap)')
io.show()
#A list of color maps can be found here: [Matplotlib color maps](https://matplotlib.org/stable/tutorials/colors/colormaps.html).
# %% Ex. 6 - Color maps

color_maps = [
    'cool',
    'hot',
    'pink',
    'copper',
    'coolwarm',
    'cubehelix',
    'terrain',
]

for map in color_maps:
    io.imshow(im_org, cmap=map)
    plt.title(f'Metacarpal image (with "{map}" colormap)')
    io.show()

# %% Ex. 7 - Grey scale scaling


io.imshow(im_org, vmin=20, vmax=170)
plt.title('Metacarpal image (with gray level scaling)')
io.show()
#Pixels with values of 20 and below will be visualized black and pixels with values of 170 and above as white and values in between as shades of gray.



def normalize_pixels(img):
    vmin = img.flatten().min()
    vmax = img.flatten().max()
    io.imshow(img, vmin=vmin, vmax=vmax)
    plt.title('Normalised image')
    io.show()

normalize_pixels(im_org)


# %% Ex. 8 - Histograms

plt.hist(im_org.ravel(), bins=256)
plt.title('Image histogram')
io.show()
h = plt.hist(im_org.ravel(), bins=256)
bin_no = 100
count = h[0][bin_no]
print(f"There are {count} pixel values in bin {bin_no}")
bin_left = h[1][bin_no]
bin_right = h[1][bin_no + 1]
print(f"Bin edges: {bin_left} to {bin_right}")

# Alternative way to call histogram:
y, x, _ = plt.hist(im_org.ravel(), bins=256)
# %% Ex. 9 - Find most common range of intensities
idx = np.argmax(h[0])
h[0][idx]

# %% Ex. 10 - Coordinate system

r = 100
c = 50
im_val = im_org[r, c]
print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

#*What is the pixel value at (r, c) = (110, 90) ?*

print(f"(r,c)=(110,90) -> {im_org[110,90]}")
# %% Ex. 11 - What does this operation do?

im_org[:30] = 0
io.imshow(im_org)
io.show()

# It sets the rows of the first 30 pixels to 0

mask = im_org > 150
io.imshow(mask)
io.show()

# Shows only the pixels with an intensity greater than 150.

# %% Ex. 12 - *Where are the values 1 and where are they 0?*

# If it's regarding the mask, they are 1 where 
# the intensity is greater than 150
# but 0 when it's equal to or under 150.


# %% Ex. 13 - *What does this piece of code do?*
im_org[mask] = 255
io.imshow(im_org)
io.show()

# %% Ex. 14 - New image
im_name = 'ardeche.jpg'

im_ardeche = io.imread(in_dir + im_name)

print(im_ardeche.shape)
print(im_ardeche.dtype)

# The image has 600 rows, 800 columns, 3 channels


# %% Ex. 15 - *What are the (R, G, B) pixel values at (r, c) = (110, 90)?*

im_ardeche[110, 90]
# The RGB values are: 119, 178, 238

r = 110
c = 90
im_ardeche[r, c] = [255, 0, 0]
io.imshow(im_ardeche)
# You can see the red pixel near the corner

# %% Ex. 16 - *Try to use NumPy slicing to color the upper half of the photo green.*

im_ardeche[:im_ardeche.shape[0]//2] = [0,255,0]
io.imshow(im_ardeche)
io.show()
# %% Ex. 17 - Own image

im_name = 'perfectblue.jpg'

im_org = io.imread(in_dir + im_name)

print(im_org.shape)
print(im_org.dtype)

image_rescaled = rescale(im_org, 0.25, anti_aliasing=True,
                         channel_axis=2)

# %% Ex. 18 - New dimensions

print(image_rescaled.shape)
print(image_rescaled.dtype)

# Resize instead of rescale (open aspect ratio)
image_resized = resize(im_org, (im_org.shape[0] // 4,
                       im_org.shape[1] // 6),
                       anti_aliasing=True)
print(image_resized.shape)
# %% Ex. 19 - Force 400 px in column width

im_gray = color.rgb2gray(im_org)
im_byte = img_as_ubyte(im_gray)

image_forced_400 = resize(im_org, 
                          (im_org.shape[0], 400),
                        anti_aliasing=True)
io.imshow(image_forced_400)
io.show()

# %% Ex. 19 - *Compute and show the histogram of you own image.*
h = plt.hist(im_org.ravel(), bins=256)


# %% Ex. 20 - dark vs light images

# I don't have time for this, the dark one will have most peaks in lower bins
# light images will have higher peaks in the upper bins

# %% Ex. 21 - Bright subject dark background

# You would be able to recognize the subject and background in the histogram
# The subject would show up on the right and the background
# would show up on the left

# %% Ex. 22 - Color channels

im_name = 'DTUSign1.jpg'

im_org = io.imread(in_dir + im_name)
print(im_org.shape)
print(im_org.dtype)
r_comp = im_org[:, :, 0]
io.imshow(r_comp)
plt.title('DTU sign image (Red)')
io.show()
# %% Ex. 23 - individual components
g_comp = im_org[:, :, 1]
io.imshow(g_comp)
plt.title('DTU sign image (Green)')
io.show()

b_comp = im_org[:, :, 2]
io.imshow(b_comp)
plt.title('DTU sign image (Blue)')
io.show()

# The red sign is dark in GB components since it's red...
# like the pixel values are low in those positions where
# the sign is

# The building is bright in all channels cause it's grey/white ish
# which means it's reflecting colors in the whole electromagnetic
# spectrum, thus raising the values of all three channels

# %% Ex. 24 - simple image manipulation
#We can make a black rectangle..?
im_org[500:1000, 800:1500, :] = 0
io.imshow(im_org)
io.show()

# %% Ex. 25 - Saving images
io.imsave(in_dir + 'DTUSign1-marked.PNG', im_org)
# %% Ex. 26 - blue rectangle over dtu sign

im_name = 'DTUSign1.jpg'

im_org = io.imread(in_dir + im_name)

im_org[1500:2000,2000:3000,:] = [0,0,255]
io.imshow(im_org)
io.show()
io.imsave(in_dir+'DTUSign1-marked-blue.PNG', im_org)

# %% Ex. 27 - Coloring bones blue

im_name = 'metacarpals.png'
im_org = io.imread(in_dir + im_name)

im_rgb = color.gray2rgb(im_org)

threshold = 120
mask = im_org > threshold
im_rgb[mask] = [0,0,255]
io.imshow(im_rgb)
io.show()
# %% Ex. 28 - *What do you see - can you recognise the inner and outer borders of the bone-shell in the profile?*
im_name = 'metacarpals.png'
im_org = io.imread(in_dir + im_name)
p = profile_line(im_org, (342, 77), (320, 160))
plt.plot(p)
plt.ylabel('Intensity')
plt.xlabel('Distance along line')
plt.show()

# The two peaks are the shells of the bone.

in_dir = "data/"
im_name = "road.png"
im_org = io.imread(in_dir + im_name)
im_gray = color.rgb2gray(im_org)
ll = 200
im_crop = im_gray[40:40 + ll, 150:150 + ll]
xx, yy = np.mgrid[0:im_crop.shape[0], 0:im_crop.shape[1]]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xx, yy, im_crop, rstride=1, cstride=1, cmap=plt.cm.jet,
                       linewidth=0)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# %% Ex. 29 - DICOM images

in_dir = "data/"
im_name = "1-442.dcm"
ds = dicom.dcmread(in_dir + im_name)
print(ds)

# The size of the image is 512x512, and can be seen in above print.

im = ds.pixel_array

# %% Ex. 30 - *Try to find the shape of this image and the pixel type? Does the shape match the size of the image found by inspecting the image header information?*

print(im.shape)
print(im.dtype)

# Yes the shape matches with the header data.

io.imshow(im, vmin=-1000, vmax=1000, cmap='gray')
io.show()

vmin = im.ravel().min()
vmax = im.ravel().max()
io.imshow(im, vmin=vmin, vmax=vmax, cmap='gray')
plt.title('Normalised image / max contrast?')
io.show()
# Well it could be better but yh. Find darkest part of the organ and brightest part.
# Then use those
io.imshow(im, vmin=-500, vmax=vmax, cmap='gray')
io.show()

# TODO: Return to Exercise 13?


# %%
