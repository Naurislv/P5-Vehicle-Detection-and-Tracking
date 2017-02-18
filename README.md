# P5-Vehicle-Detection-and-Tracking
Fith project submission at Udacity Self Driving Nanodegree program - Vehicle detecting and tracking using clasic machine learning algoritms and computer vision

# Udacity Self Driving Nanodegree 5. Project "Vehicle Detection and Tracking"

## Project submission

Resulted project video link : https://www.youtube.com/watch?v=wrO1LYdThLE&feature=youtu.be

Training datasets can be downloaded :

* cars : https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
* not cars : https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip

All code can be found in Nauris_Dorbe_P5.ipynb

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

[//]: # (Image References)

[image1]: ./vehicle_notvehicle_example.jpg "example"
[image2]: ./hog_features.jpeg "HOG"
[image3]: ./test_image_windows.jpg "sliding_windows"
[image4]: ./test_image_detectedcars.jpg "detected_cars"
[image5]: ./image_final_detection.jpeg "final_detection"


### Code structure

#### Classes : 

* Utils : some useful functions
* detect_cars : pipeline for detecting, tracking and drawing bounding boxes around cars in image stream

#### Functions :

* get_hog_features : perform Histogram of Oriented Gradients feature extraction from image
* bin_spatial : extract binned color features
* color_hist : extract color histogram features
* extract_features : perform previous functions to create feature vector
* single_img_features : very similar to extract_features function, but for single image rather than list of images
* slide_window : slide window across image to cut smaller images for classification
* search_windows : use classifier to find in which image (after slide_window) car is represented.
* draw_boxes : draw bounding boxes in image

---

### Pipeline

#####  Breaf description of pipeline used to perform car detection and tracking in image stream from video.

Examples of training data used in project. Left 'vehicle', right 'not vehicle'.

![alt text][image1]

### Histogram of Oriented Gradients (HOG)

To extract HOG features I use skimage 'from skimage.feature import hog' library. What this library actually does is compute gradient magnitue and directions for each pixel in image and then group them by use defined blocks. Then for each block histogram of direction magnitudes are computed, so more dominante direction with most magnitude will have highest histogram values. It gives calssifier information about image edge structor. Note that hog only accepts one channel image.

In image below are examples of HOG features from image covrted to LUV channels. Few things should be noted - you can easily spot that there are mayor differences between channels. For example car image have second and third channels different HOG features but for not-car image HOG feature look similar for second and third channel. Also in example you can see different parameter 'pixels per cell'. Where there are less pixels per cell you can see that it's easily to spot such objects such as car lights but where there are more pixels per cell its easier to spot car body itself.

For this project after longs testing itterations I decided that following parameters for HOG worked best :

1. LUV first channel -> all three channels gave better accuracy but also resulted in much more computing time
2. orientation bins : 8
3. pixels per cell : 8
4. cell per block : 2

It is worth noting that we could extract all three LUV channels, maybe also RGB channels, YCrCb channels and combine them but that would increase computing time drastically, so those parameters are balance between computing time and test set accuracy.

Example of how to extract HOG features from LUV first channel:

```
vehicle_example_LUV = cv2.cvtColor(vehicle_example, cv2.COLOR_RGB2LUV)[:, :, 0]

orient = 8
pix_per_cell = 8
cell_per_block = 2

hog_features = get_hog_features(vehicle_example_LUV, orient, pix_per_cell,
                                cell_per_block, vis=False, feature_vec=True)
```

![alt text][image2]

### Sliding Window Search

For sliding window slide_window function were used. In total 130 windows was enough to get enough windows for classification so the right vehicle tracking would happen. As for project only certain part of image was defined for slide_window function, x in interval 650:im_width and y from 370:im_hight. I use four different sliding window sizes and 4 different y startin:ending points to get result needed (balance between performance and speed). First window 96x96 was defined for car which appear to be very small thus far away, but it also helped to locate parts of bigger cars. Second window 110x145 medium sized cars. Third window 155x165 big cars. Fourt and last window 170x200 was used to detect starting point of car where it starting to appear in image. But it is worth noting that all windows helped to allocate some part of car to get as precise location as needed.

Example of how to perform sliding window across image region:

```
test_image_path = 'test_images/test1.jpg'
test_image = cv2.imread(test_image_path)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

x_start_stops = [[650, None], [650, None], [650, None], [650, None]] # Min and max in y to search in slide_window()
xy_windows = [(96, 96), (110, 145), (155, 165), (170, 200)] # x, y

y_start_stops = [[395, 530], [380, 590], [370, 590], [400, 700]] # Min and max in y to search in slide_window()
xy_overlaps = [(0.9, 0.5), (0.7, 0.6), (0.6, 0.5), (0.6, 0.5)] # x, y

windows = []
for y_start_stop, x_start_stop, xy_window, xy_overlap in zip(y_start_stops, x_start_stops, xy_windows, xy_overlaps):
    windows += slide_window(test_image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                            xy_window=xy_window, xy_overlap=xy_overlap)
    
test_image_windows = draw_boxes(test_image, windows, color=(0, 0, 255), thick=4)
```

![alt text][image3]

### Vehicle detection in image

As previously described to detected cars in images HOG features are used in certain way. Also LUV color space used as described before. That and spatially binned color features with size of 16x16 plus color histogram features with 16 histogram bins and color values (0, 256) after testing proved to give best balance between speed and performance.

Before calssifier feature vector is normalized using sklearn StandardScaler().

I used Linear Support Vector Classification 'from sklearn.svm import LinearSVC' with default parameters which proved to be precise and fast enough for project video. Dataset before classifier were divided into two train, test datasets and shuffled for better classifier performance.

Feature vector size is 2384 which consisted of HOG, binned color and color histogram features. To test classifier performance I tested it on 3552 testing samples on which I got 0.9885 accuracy. And then to test speed of classifier I tested it on same 3552 testing samples to run 100 loops and measure 3 best times per loop which is 10.4 ms.

```
color_space = 'LUV'
spatial_size = (16, 16)
hist_bins = 16 # Number of histogram bins
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"

spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

car_features = extract_features(cars_training_set, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars_training_set, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
    
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)

hot_windows = search_windows(test_image, windows, svc, X_scaler, color_space=color_space, 
                             spatial_size=spatial_size, hist_bins=hist_bins, 
                             orient=orient, pix_per_cell=pix_per_cell, 
                             cell_per_block=cell_per_block, 
                             hog_channel=hog_channel, spatial_feat=spatial_feat, 
                             hist_feat=hist_feat, hog_feat=hog_feat)

test_image_DetectedCars = draw_boxes(test_image, hot_windows, color=(0, 0, 255), thick=4)

U.pim([test_image_DetectedCars])

save_im = cv2.cvtColor(test_image_DetectedCars, cv2.COLOR_RGB2BGR)
s = cv2.imwrite("test_image_detectedcars.jpg", save_im)
```
![alt text][image4]

In the end I created class 'detect_cars' which I used to detect, find and track cars. When initializing if test=True it performs single image car detection. After finding windows in image heat map was computed and final bounding box found using scipy label function and then draw on actual image. Below is code example of 'detect_cars' usage and test image examples. To remove false-positives I used 'apply_threshold' with threshold 1 or in other words needed at least two windows to overlap to count as a detection. With that I remove all false positives.

```
D = detect_cars() # initilize detect_cars class

for i in range(1, 7):
    image_path = 'test_images/test{}.jpg'.format(i)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    D.find_cars(image, test=True)
```

![alt text][image5]

---

### Video Implementation

Resulted video link : https://www.youtube.com/watch?v=wrO1LYdThLE&feature=youtu.be

To get more stable results in video I recorded position of previously detected bounding boxes and then for each new bounding box I compared it with prevously detected if distance was below defined threshold then match is found and new bounding box coordinated were slighty adjusted based on prevous bounding box - this prevented the bounding box to bounce around which would seem to be unstable. This was performed by 'adjust_bbox' function in class 'detect_cars'. Also sometimes for some frames bounding box were not found at all and such cases previous bounding box were used but if that happend for user defined times then also that is dopped.

Sample code to perform vehicle tracking on video :

```
D = detect_cars() # initilize detect_cars class

# Need to initilize lane_finding class before start using pipeline otherwise pipeline will use previously saved values
output = 'test_video_done.mp4'
clip1 = VideoFileClip("test_video.mp4")
done_clip = clip1.fl_image(D.find_cars) #NOTE: this function expects color images!!
%time done_clip.write_videofile(output, audio=False)
```

---

## Discussion

1. Main  problem I faced was parameter tuning and in this project parameters are quite much such as color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_chells, window size, window locations, classifier and so on. To solve this we should create some kind of testing system, which could test various combinations of parameters and choose the best one. This, however, might be very difficult for slinding window and also for defining which result is best because we did not have training set here and classifier accuracy may not be the best metric due to overfitting and other related issues. Other way to solve this issue would be to choose different approach such as Fully Convolution Neural networks.

2. Other issue of-course is speed which is not great for whole system. Classifier itself worked relatively fast but feature extraction, sliding window etc. could do better. This again could be addressed with Fully Convolution Neural networks.

3. My solution may not work for detecting cars which are driving towrds camera because classifier trained only on images of car backs. This could be solved to train on bigger dataset with images with front view of car.
