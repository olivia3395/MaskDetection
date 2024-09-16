# **Mask Detection with Deep Learning**

## **Overview**

In response to the global COVID-19 pandemic, wearing masks has become one of the most crucial measures to curb the virus's spread. This project aims to create a robust, automated **mask detection system** using **deep learning** models that can accurately detect whether a person is wearing a mask or not. The system can be integrated into security setups in public areas such as **airports**, **hospitals**, **malls**, and **offices**, ensuring compliance with safety protocols in real-time.

The project employs popular deep learning architectures like **MobileNetV2** and **ResNet**, pre-trained on **ImageNet**, and fine-tunes them for binary classification: detecting "with mask" or "without mask".

---

## **Project Goals**

1. **Develop an Effective Mask Detection Model**: The objective is to leverage transfer learning from pre-trained models like MobileNetV2 and ResNet to detect mask usage efficiently and accurately.
   
2. **Enhance Model Generalization**: By applying data augmentation techniques, the model is trained to generalize well under various conditions (e.g., different lighting, angles, and mask styles).

3. **Real-World Deployment**: The model is optimized for performance and size, making it suitable for deployment on edge devices (e.g., security cameras or mobile devices) for real-time detection.

---

## **Dataset**

### **Data Collection**
The dataset consists of a large number of labeled images, split into three categories:
- **With Mask**: Images of individuals correctly wearing masks.
- **Without Mask**: Images of individuals without masks.
- **Improper Mask** (optional): Images where masks are worn improperly (optional but useful to train the model for nuanced classifications).

The dataset was split into **training**, **validation**, and **test** sets to ensure a robust evaluation process.

### **Data Augmentation**
To improve the generalization capabilities of the model, various data augmentation techniques were applied:
- **Random Rotation**: To simulate different head positions.
- **Horizontal Flipping**: To add variability to image orientations.
- **Color Jitter**: To simulate different lighting conditions and improve the model’s robustness in real-world environments.
- **Resizing**: All images were resized to match the input size expected by the deep learning models (e.g., 224x224 for MobileNetV2).

These augmentation techniques help the model adapt to various conditions, making it more reliable in real-world scenarios.

---

## **Model Architecture**

### **Pre-Trained Models**
We utilized the following **pre-trained models**:
1. **MobileNetV2**: Known for its efficiency and lightweight architecture, making it suitable for real-time applications.
2. **ResNet**: A deeper architecture that achieves high accuracy, especially in scenarios requiring more nuanced feature extraction.

These models were trained on the **ImageNet** dataset and transferred to the task of mask detection by replacing their final layers with a binary classification head (mask vs. no mask).

### **Transfer Learning**
Instead of training models from scratch, we applied **transfer learning** by freezing the early layers of the pre-trained models. This allowed us to retain the robust feature extraction capabilities of these models while fine-tuning the last few layers to adapt to the specific task of mask detection.

---

## **Training Strategy**

### **Loss Function & Optimizer**
The model was trained using a **cross-entropy loss function**, which is suitable for multi-class classification tasks. To optimize the model weights, we employed the **Adam optimizer** due to its efficiency in handling large datasets and its adaptive learning rate capabilities.

### **Training Process**
- **Epochs**: The model was trained over several epochs, gradually improving its accuracy with each pass through the dataset.
- **Batch Size**: A batch size was chosen that balanced computational efficiency and convergence speed, ensuring the model could process enough data without overloading memory.

During the training process, key metrics such as **accuracy**, **precision**, **recall**, and **F1 score** were monitored to ensure the model was improving and to prevent overfitting.

### **Validation and Test**
After each epoch, the model was validated on a separate dataset (the **validation set**) to ensure that it was not overfitting and could generalize well to unseen data. Final performance metrics were calculated on a **test set**, which the model had never seen before, providing an unbiased assessment of its real-world performance.

---

## **Model Evaluation**

### **Accuracy and Loss Monitoring**
- **Training Accuracy & Loss**: These were tracked to monitor how well the model was learning from the training data.
- **Validation Accuracy & Loss**: Used to track the model's performance on unseen data after each training epoch, helping to identify overfitting.
- **Test Accuracy**: The ultimate metric to evaluate the generalizability of the model to completely unseen data.

### **Confusion Matrix**
A confusion matrix was generated to assess where the model was making errors—whether it was incorrectly classifying "mask" as "no mask" or vice versa. This helped identify common patterns in the errors, which could guide further improvements.

### **Class Activation Maps (Grad-CAM)**
**Grad-CAM** (Gradient-weighted Class Activation Mapping) was employed to visualize which parts of the image the model was focusing on when making predictions. This helped ensure that the model was learning the correct features (such as focusing on the mouth/nose region to detect masks).

---

## **Results**

The trained model achieved high accuracy across all datasets, showing effective performance in both detecting masks and differentiating between masked and unmasked individuals. Key results include:
- **Training Accuracy**: Over 95% accuracy on the training set.
- **Validation Accuracy**: Consistent performance with over 94% accuracy on the validation set.
- **Test Accuracy**: The final model reached over 93% accuracy on the test set, demonstrating strong generalization capabilities.

The confusion matrix and Grad-CAM visualizations indicated that the model was focusing correctly on the key areas (i.e., the face) and performed well even in challenging conditions (e.g., occlusions or poor lighting).

---


### **Acknowledgments**

Special thanks to the open-source community and pre-trained model resources like **ImageNet**, which significantly reduced the time and effort required to build this solution.
