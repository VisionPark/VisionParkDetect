# VisionParkDetect
## Summary
This repository contains the implementation of a parking space occupancy classifier, which utilizes a vision algorithm to classify parking spaces as either free or occupied. The algorithm employs classical Computer Vision techniques, including edge detection and background subtraction, to process the segmentation of parking spaces.

![imagen](https://github.com/VictorNarov/VisionParkDetect/assets/19890871/1fe70316-9693-4efb-b6d4-cfd57efa5c21)

## Metrics
An extensive study of the input parameters has been conducted to determine a valid configuration for each parking lot in the validation set. This study resulted in the generation of ROC performance curves and evaluation metrics such as sensitivity, specificity, precision, and F1 score.

![imagen](https://github.com/VictorNarov/VisionParkDetect/assets/19890871/22c8c0ed-2cf1-4b09-a1e0-dbeb374cacbe)

## Dataset
The classifier is trained and validated using the PKLot dataset, which consists of 12,417 images from three different parking lots captured by surveillance cameras. The dataset includes images taken under three environmental conditions: sunny, cloudy, and rainy. Each parking space in the dataset is labeled as either free or occupied, and the corresponding vertices defining the space's segmentation are provided. The parking lots contain 100, 28, and 40 parking spaces, respectively.
The training process utilized 80% of the dataset, while the remaining 20% was reserved for validation. The parameter study was further divided based on parking lot and weather condition, resulting in a total of nine subsets for training and validation.

![imagen](https://github.com/VictorNarov/VisionParkDetect/assets/19890871/b6433ea8-a896-48e9-9fe8-0ffdcd55fb60)

## Results
After conducting experiments and evaluations, the vision algorithm's parameters were fine-tuned for each subset, achieving impressive precision and sensitivity ranging from 82% to 97%.

![imagen](https://github.com/VictorNarov/VisionParkDetect/assets/19890871/8582d555-f9f6-4d60-ac11-b12cad246f9d)

This project demonstrates the successful implementation of a parking space occupancy classifier, offering valuable insights into the intersection of Computer Vision and parking management systems. The repository serves as a valuable resource for researchers, developers, and practitioners interested in enhancing the efficiency and sustainability of urban parking systems.
