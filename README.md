# VisionParkDetect
## Summary
This repository contains the implementation of a parking space occupancy classifier, which utilizes a vision algorithm to classify parking spaces as either free or occupied. The algorithm employs classical Computer Vision techniques, including edge detection and background subtraction, to process the segmentation of parking spaces.

![imagen](https://github.com/VisionPark/VisionParkDetect/assets/19890871/1a21c378-22b3-4b43-ba82-493e42fc8d39)


## Metrics
An extensive study of the input parameters has been conducted to determine a valid configuration for each parking lot in the validation set. This study resulted in the generation of ROC performance curves and evaluation metrics such as sensitivity, specificity, precision, and F1 score.

![imagen](https://github.com/VisionPark/VisionParkDetect/assets/19890871/aa4e8727-742c-43b0-b7b4-e6ebc3beefe2)


## Dataset
The classifier is trained and validated using the PKLot dataset, which consists of 12,417 images from three different parking lots captured by surveillance cameras. The dataset includes images taken under three environmental conditions: sunny, cloudy, and rainy. Each parking space in the dataset is labeled as either free or occupied, and the corresponding vertices defining the space's segmentation are provided. The parking lots contain 100, 28, and 40 parking spaces, respectively.
The training process utilized 80% of the dataset, while the remaining 20% was reserved for validation. The parameter study was further divided based on parking lot and weather condition, resulting in a total of nine subsets for training and validation.

![imagen](https://github.com/VisionPark/VisionParkDetect/assets/19890871/d7153dbc-7285-456e-b4c2-50d964921fd9)


## Results
After conducting experiments and evaluations, the vision algorithm's parameters were fine-tuned for each subset, achieving impressive precision and sensitivity ranging from 82% to 97%.

![imagen](https://github.com/VisionPark/VisionParkDetect/assets/19890871/50726e73-bbdc-43dd-86bb-55efa152ee1c)


This project demonstrates the successful implementation of a parking space occupancy classifier, offering valuable insights into the intersection of Computer Vision and parking management systems. The repository serves as a valuable resource for researchers, developers, and practitioners interested in enhancing the efficiency and sustainability of urban parking systems.
