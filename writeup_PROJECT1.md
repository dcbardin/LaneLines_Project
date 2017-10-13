# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

[image2]: ./test_images/solidWhiteCurve_blur.png "Gassian"

[image3]: ./test_images/solidWhiteCurve_canny.png "Canny on original image withut noise"

[image4]: ./test_images/solidWhiteCurve_masked_edges.png "Region of interest"

[image5]: ./test_images/solidWhiteCurve_lines_edges.png "Region of interest"

---

### Reflection

### 1. Import pacages and load image

I import the respective packages and load a test image. Matplotlib read image as RGB and cv2 read the iage as GBR. 


```
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import imageio
import math
imageio.plugins.ffmpeg.download()
%matplotlib inline
import os
input_images = os.listdir("test_images/")

image = cv2.imread('test_images/solidWhiteCurve.jpg')
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)
gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
plt.imshow(gray, cmap='gray')
```





### 2. Convert image to gray and remove the noises

To remove the noises from the gray image, it was appled cv2.GaussianBlur


```
# Gaussian smoothing
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
plt.imshow(blur_gray)
mpimg.imsave("Test_images_output/solidWhiteCurve_blur.png", blur_gray)
```

![alt text][image2]

### 3. Apply Canny edges

Canny edges method is applied t find all the lines in images Without noise, the edges of lane line is very clear.

```
# Canny Edge
low_threshold = 50
high_threshold = 150

edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
plt.imshow(edges)
mpimg.imsave("Test_images_output/solidWhiteCurve_canny.png", edges)
```

### 4. Define the interested area

In this case it was applied the np.array to define the interested area and cv2.fillPoly to filling the pixels with color

```

#defining a blank mask
mask = np.zeros_like(edges)

if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
else:
        ignore_mask_color = 255

imshape = image.shape   

vertices = np.array([[(100,imshape[0]), (440, 325), (550, 325), (imshape[1],imshape[0])]], dtype=np.int32)
      
#filling pixels inside the polygon defined by "vertices" with the fill color    
cv2.fillPoly(mask, vertices, ignore_mask_color)

#returning the image only where mask pixels are nonzero
masked_edges = cv2.bitwise_and(edges, mask)

plt.imshow(masked_edges)
mpimg.imsave("Test_images_output/solidWhiteCurve_masked_edges.png", masked_edges)

```

![alt text][image3]


### 6. Generate HoughLines

Function used to remove short and useless edges (Input parameter)

```
# Hough Transform
rho = 3 # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15 # minimum number of votes (intersections in Hough grid cell)
min_line_length = 5 #minimum number of pixels making up a line
max_line_gap = 25 # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

```


### 7. Apply lines on the Image

```
left_x = []
left_y = []
right_x = []
right_y = []

sizeX = image.shape[1]
sizeY = image.shape[0]
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(x1-x2) == 0:
        slope = float("inf")
    else:
        slope = (y2 - y1)/(x2 - x1)

    if slope > -0.5 and x1 > sizeX/2 and x2 > sizeX/2:
        right_x.append(x1)
        right_x.append(x2)
        right_y.append(y1)
        right_y.append(y2)
    elif slope < 0.5 and x1 < sizeX/2 and x2 < sizeX/2:
        left_x.append(x1)
        left_x.append(x2)
        left_y.append(y1)
        left_y.append(y2)

r, r_b = np.polyfit(right_x, right_y, 1)
l, l_b = np.polyfit(left_x, left_y, 1)

y1 = image.shape[0]
y2 = image.shape[0] * (1 - 0.35)
x1r = (y1 - r_b) / r
x2r = (y2 - r_b) / r
x1l = (y1 - l_b) / l
x2l = (y2 - l_b) / l

color = [255,0,0]
thickness = 10
cv2.line(image, (int(x1r), y1), (int(x2r), int(y2)), color, thickness)
cv2.line(image, (int(x1l), y1), (int(x2l), int(y2)), color, thickness)

color_edges = np.dstack((edges, edges, edges)) 
lines_edges = cv2.addWeighted(image, 1, line_image, 1, 0) 
plt.figure()
plt.imshow(lines_edges)
mpimg.imsave("Test_images_output/solidWhiteCurve_lines_edges.png", gray)
```


![alt text][image5]


### 8. Apply lines on the Video

In this case, it was necessary apply the Pipeline inside a Function Process_image()

```
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
```

### 9. Identify potential shortcomings

Shortcoming:
In the challenge.mp4, it was not possible to apply current pipelie, since it applies liner equation and do not works in curve lines. I tried some codes, but did not work.
Current pipe line has a simple image treatment and limited parameters, and it can fail in other condition of images (Rain, Sands, Fog, etc). The pipeline must adapt for all kind of situation.


### 10. Improvements for Pipeline
- In paralell with next classes I will try improve pipeline to works in challenge movies (Curves condition) and try understand how to adapt the pipeline with outher condition of images (Rain, sands, fog, etc).

- In current movie (Linear lines), some frames is not perfect, with abnormal movement of lines, I need improve this case as well.
