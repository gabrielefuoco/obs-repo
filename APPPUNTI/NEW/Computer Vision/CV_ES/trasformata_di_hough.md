# Hough transform

```python
import os
import numpy as np
import cv2 # OpenCV-Python
%matplotlib inline
import matplotlib.pyplot as plt
import imageio
from skimage.color import rgb2gray, gray2rgb
import skimage

plt.rcParams['figure.figsize'] = (12, 8)

print("OpenCV-Python Version %s" % cv2.__version__)
```

    OpenCV-Python Version 4.5.1
    

```python
building = np.array(imageio.imread('build_001.jpg'))

plt.imshow(building);
```

    
![png](trasformata_di_hough_2_0.png)
    

```python
gray_image = rgb2gray(building)

plt.imshow(gray_image, cmap='gray');
```

    
![png](trasformata_di_hough_3_0.png)
    

```python
cv2.Canny?
```

.   @brief Finds edges in an image using the Canny algorithm @cite Canny86 .
.   
.   The function finds edges in the input image and marks them in the output map edges using the
.   Canny algorithm. The smallest value between threshold1 and threshold2 is used for edge linking. The
.   largest value is used to find initial segments of strong edges. See

    
Canny(dx, dy, threshold1, threshold2[, edges[, L2gradient]]) -> edges
.   \overload
.   
.   Finds edges in an image using the Canny algorithm with custom image gradient.
.   

```python
img = skimage.img_as_ubyte(gray_image)
edges = cv2.Canny(img, 50, 200, None, 3)

plt.imshow(edges, cmap='gray');
```

    
![png](trasformata_di_hough_5_0.png)
    

```python
cv2.HoughLines?
```

@brief Finds lines in a binary image using the standard Hough transform.
.   
.   The function implements the standard or standard multi-scale Hough transform algorithm for line
.   detection. See <http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm> for a good explanation of Hough
.   transform.
.   

```python
lines = cv2.HoughLines(edges, 1, np.pi / 180, 400, None, 0, 0)

lines.shape
```

## Polar coordinates

ρ = x cos θ + y sin θ

where:

ρ (rho) = distance from origin to the line. [-max_dist to max_dist].
          max_dist is the diagonal length of the image.  
θ = angle from origin to the line. [-90° to 90°]

```python
def compute_line_parameters(point1, point2):
    # ax + by = c
    # m = -a/b   n = c/b
    a = point2[1] - point1[1]
    b = point1[0] - point2[0]
    c = a*(point1[0]) + b*(point1[1])
    if a != 0 and b != 0:
        return [-a/b, c/b]

plotted_lines = []

for i in range(lines.shape[0]):
    rho = lines[i, 0, 0]
    theta = lines[i, 0, 1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a)))
    pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a)))
    
    line = compute_line_parameters(pt1, pt2)
    if line:
        plotted_lines.append(line)
        
print('lines', len(plotted_lines))

plt.figure(figsize=(29,29))
plt.xlim([-2500, 7500])
plt.ylim([7500,-7500])
plt.imshow(building)
for line in plotted_lines:
    f = lambda x: line[0]*x + line[1]
    x = np.linspace(-2500, 7500)
    y = f(x)
    plt.plot(x, y, 'r')
plt.show()
```

    lines 31
    

    
![png](trasformata_di_hough_9_1.png)
    

# Probabilistic Hough

```python
cv2.HoughLinesP?
```

@brief Finds line segments in a binary image using the probabilistic Hough transform.
.   
.   The function implements the probabilistic Hough transform algorithm for line detection, described

```python
lines_p = cv2.HoughLinesP(edges, 1, np.pi / 180, 400)

lines_p.shape
```

    (446, 1, 4)

```python
building_copy = np.copy(building)

for i in range(lines_p.shape[0]):
    pt1 = (lines_p[i, 0, 0], lines_p[i, 0, 1])
    pt2 = (lines_p[i, 0, 2], lines_p[i, 0, 3])
    
    cv2.line(building_copy, pt1, pt2, (255,0,0), 5)

plt.imshow(building_copy)
plt.show()
```

    
![png](trasformata_di_hough_13_0.png)
    

```python
def compute_line_parameters(point1, point2):
    # ax + by = c
    # m = -a/b   n = c/b
    a = point2[1] - point1[1]
    b = point1[0] - point2[0]
    c = a*(point1[0]) + b*(point1[1])
    if a != 0 and b != 0:
        return [-a/b, c/b]

plotted_lines = []

for i in range(lines_p.shape[0]):
    pt1 = (lines_p[i, 0, 0], lines_p[i, 0, 1])
    pt2 = (lines_p[i, 0, 2], lines_p[i, 0, 3])
    
    line = compute_line_parameters(pt1, pt2)
    if line:
        plotted_lines.append(line)
        
print('lines', len(plotted_lines))

plt.figure(figsize=(29,29))
plt.xlim([-2500, 7500])
plt.ylim([7500,-7500])
plt.imshow(building)
for line in plotted_lines:
    f = lambda x: line[0]*x + line[1]
    x = np.linspace(-2500, 7500)
    y = f(x)
    plt.plot(x, y, 'r')
plt.show()
```

    lines 176
    

    
![png](trasformata_di_hough_14_1.png)
    

# Detect Cyrcle

```python
coin_colored = np.array(imageio.imread('coin.jpg'))

coin = rgb2gray(coin_colored)

plt.imshow(coin, cmap='gray');
```

    
![png](trasformata_di_hough_16_0.png)
    

```python
#Parameters like minDist, minRadius, maxRadius can vary from image to image
img = skimage.img_as_ubyte(coin)
circles_float = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 0.9, minDist=120, param1=50, param2=30, minRadius=90, maxRadius=220)
circles = np.uint16(np.around(circles_float))
print(circles.shape)
```

    (1, 138, 3)
    

```python
coin_blur = cv2.GaussianBlur(coin, (31, 31), 5)

plt.imshow(coin_blur, cmap='gray');
```

    
![png](trasformata_di_hough_18_0.png)
    

```python
img = skimage.img_as_ubyte(coin_blur)
circles_float = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 0.9, minDist=120, param1=50, param2=30, minRadius=90, maxRadius=220)
circles = np.uint16(np.around(circles_float))
print(circles.shape)
```

    (1, 13, 3)
    

```python
circles
```

    array([[[ 868,  282,  184],
            [ 476,  928,  173],
            [1200, 1634,  183],
            [ 682, 2470,  183],
            [1888, 1148,  167],
            [ 696, 3026,  155],
            [ 728, 3572,  195],
            [1858, 1670,  159],
            [2450,  358,  160],
            [2644, 3030,  175],
            [1870, 2818,  142],
            [1888, 2280,  169],
            [2628, 3488,  162]]], dtype=uint16)

```python
cv2.HoughCircles?
```

 @brief Finds circles in a grayscale image using the Hough transform.
    .   
    .   The function finds circles in a grayscale image using a modification of the Hough transform.
    .   

```python
# sort by x-coord
circles = np.squeeze(circles)
circles = circles[ circles[:,0].argsort()]
print(circles)
print(circles.shape)
```

    [[ 476  928  173]
     [ 682 2470  183]
     [ 696 3026  155]
     [ 728 3572  195]
     [ 868  282  184]
     [1200 1634  183]
     [1858 1670  159]
     [1870 2818  142]
     [1888 1148  167]
     [1888 2280  169]
     [2450  358  160]
     [2628 3488  162]
     [2644 3030  175]]
    (13, 3)
    

```python
img_coin = np.copy(coin_colored)

for i in range(circles.shape[0]):
    c = (circles[i, 0], circles[i, 1])
    r = circles[i, 2]
    cv2.circle(img_coin, c, r, (0,0, 255), 10)
    
plt.rcParams["figure.figsize"] = (25,15)    
plt.imshow(img_coin);
```

    
![png](trasformata_di_hough_23_0.png)
    

