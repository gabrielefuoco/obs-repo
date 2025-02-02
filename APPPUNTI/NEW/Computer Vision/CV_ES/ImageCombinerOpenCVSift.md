# Applicazione di Sift per combinare 2 immagini

```python
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

%matplotlib inline
```

Obiettivo è quello di combinare due immagini in un'unica immagine facendo combiaciare le parti corrrispondenti

La strategia è quella di individuare i keypoints, trovare le corrispondenze tra l'immagine A e quella B e poi effettuare le opportune trasformazioni per effettuare il merge

Utilizziamo l'implementazione OpenCV

```python
SIFT = cv2.xfeatures2d.SIFT_create()
```

```python
# Images

image_left = cv2.cvtColor(cv2.imread('mountain_view1.png'), cv2.COLOR_BGR2RGB)
image_right = cv2.cvtColor(cv2.imread('mountain_view2.png'), cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes = axes.ravel()

for i, mx in enumerate((image_left, image_right)):
    axes[i].imshow(mx)
    axes[i].axis('off')
         
plt.show()
```

    
![png](ImageCombinerOpenCVSift_4_0.png)
    

## Analizziamo l'immagine A

```python
image_left_gray = cv2.cvtColor(image_left, cv2.COLOR_RGB2GRAY)

keypoints_left, descriptors_left = SIFT.detectAndCompute(image_left_gray, None)

len(keypoints_left)
```

    2233

```python
img_2 = cv2.drawKeypoints(image_left_gray, keypoints_left, image_left.copy())

plt.figure(figsize=(16, 12))
plt.imshow(img_2);
```

    
![png](ImageCombinerOpenCVSift_7_0.png)
    

## Analizziamo l'immagine B

```python
image_right_gray = cv2.cvtColor(image_right, cv2.COLOR_RGB2GRAY)

keypoints_right, descriptors_right = SIFT.detectAndCompute(image_right_gray, None)

len(keypoints_right)
```

    2364

```python
img_2_right = cv2.drawKeypoints(image_right_gray, keypoints_right, image_right.copy())

plt.figure(figsize=(16, 12))
plt.imshow(img_2_right);
```

    
![png](ImageCombinerOpenCVSift_10_0.png)
    

## Combiniamo le immagini

1. Si individuano le corrispondenze tra i keypoint di A e di B
2. Si trasforma l'immagine B e si effettua il merge

Per individuare la corrispondenza si può utilizzare un algoritmo basato sul criterio di distanza euclidea per individuare le coppie di keypoints $(kp_A, kp_B)$.

Per ogni punto si individuano i *k* punti *più vicini* al punto candidato. Si itera finché non si raggiunge una configurazione ottimale o si raggiunge il numero massimo di iterazioni.

Al termine del processo, si può effettuare una raffinamento del risultato eliminando le coppie con un valore di distanza *elevato*.

```python
ratio = .75

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

matches = bf.knnMatch(descriptors_right, descriptors_left, k=2)

good = []
# loop over the raw matches
for m,n in matches:
    # ensure the distance is within a certain ratio of each
    # other (i.e. Lowe's ratio test)
    if m.distance < n.distance * ratio:
        good.append(m)
matches = np.asarray(good)

matches.shape
```

    (507,)

visualizziamo la corrispondenza di 100 punti tra tutti quelli individuati

```python
matches_sublist = np.random.choice(matches.flatten(), 100)

img_desc = cv2.drawMatches(image_right, keypoints_right, image_left, keypoints_left, 
                           matches_sublist,
                           None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    

plt.figure(figsize=(20, 12))
plt.imshow(img_desc)
plt.axis('off')
plt.show()
```

    
![png](ImageCombinerOpenCVSift_14_0.png)
    

Per alli

```python
def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        return H, status
    else:
        raise RuntimeError('Can’t find enough keypoints.')
    
    
H, status = getHomography(keypoints_right, keypoints_left, 
                          descriptors_right, descriptors_left,
                          matches, 3)
    
H    
```

```python
# Apply panorama correction
width = image_left.shape[1] + image_right.shape[1]
height = image_left.shape[0] + image_right.shape[0]

result = cv2.warpPerspective(image_right, H, (width, height))
#result[0:image_right.shape[0], 0:image_right.shape[1]] = image_right
result[0:image_left.shape[0], 0:image_left.shape[1]] = image_left

plt.figure(figsize=(20,10))
plt.imshow(result)

plt.axis('off')
plt.show()
```

    
![png](ImageCombinerOpenCVSift_17_0.png)
    

```python
# transform the panorama image to grayscale and threshold it 
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

# Finds contours from the binary image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# get the maximum contour area
c = max(cnts, key=cv2.contourArea)

# get a bbox from the contour area
(x, y, w, h) = cv2.boundingRect(c)

# crop the image to the bbox coordinates
result = result[y:y + h, x:x + w]

# show the cropped image
plt.figure(figsize=(20,10))
plt.imshow(result);
```

    
![png](ImageCombinerOpenCVSift_18_0.png)
    

```python
# Images
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes = axes.ravel()

for i, mx in enumerate((image_left, image_right)):
    axes[i].imshow(mx)
    axes[i].axis('off')

    
plt.figure(figsize=(20,10))
plt.imshow(result);
    
plt.show()
```

    
![png](ImageCombinerOpenCVSift_19_0.png)
    

    
![png](ImageCombinerOpenCVSift_19_1.png)
    

