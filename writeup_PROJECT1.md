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


### 2. Convert image to gray and remove the noises

To remove the noises from the gray image, it was appled cv2.GaussianBlur

![alt text][image2]

### 3. Apply Canny edges

Canny edges method is applied t find all the lines in images Without noise, the edges of lane line is very clear.

### 4. Apply Canny edges

Canny edges method is applied t find all the

![alt text][image3]

### 5. Select intereste region
In this case, the lane lines edges is located in a trapezoid. It was defined a trapezoid to restrict the region of interest.

![alt text][image4]

### 6. Generate HoughLines

Function used to remove short and useless edges (Input parameter)

### 7. Apply lines on the Image
In this case, the lane lines edges is located in a trapezoid. It was defined a trapezoid to restrict the region of interest.


![alt text][image5]




