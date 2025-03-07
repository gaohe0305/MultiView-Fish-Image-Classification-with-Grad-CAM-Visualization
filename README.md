# MultiView-Fish-Image-Classification-with-Grad-CAM-Visualization
Multi-view fish image classification using PyTorch. Three ResNet18 models extract features from side, back, and belly views, fused via an attention mechanism for final classification. Grad-CAM visualizes model focus areas.
![图片2](https://github.com/user-attachments/assets/7f511702-5935-445b-a8c0-e6f0df13ccb8)
## Environment Requirements
<br>Python 3.6+
<br>PyTorch (recommended 1.7.0 or above)
<br>torchvision
<br>OpenCV (opencv-python)
<br>scikit-learn
<br>matplotlib
<br>pandas
<br>tqdm
<br>Pillow
<br>Install the required packages with:

<br>``` pip install torch torchvision opencv-python scikit-learn matplotlib pandas tqdm pillow ```

## Data Preparation
<br>Organize your dataset using the following structure,(You can also check the `data` folder):
<br>data/
<br>  ├── train/
<br>  │    ├── class1/
<br>  │    │      fishID_side.jpg/png/… 
<br>  │    │      fishID_back.jpg/png/…
<br>  │    │      fishID_belly.jpg/png/…
<br>  │    ├── class2/
<br>  │    │      …
<br>  │    └── ...
<br>  └── val/
<br>       ├── class1/
<br>       │      fishID_side.jpg/png/… 
<br>       │      fishID_back.jpg/png/…
<br>       │      fishID_belly.jpg/png/…
<br>       ├── class2/
 <br>      │      …
 <br>      └── ...
<br>Image Naming: Each image must include a suffix indicating its view (_side, _back, or _belly). Supported formats are .jpg, .jpeg, .png, and .bmp.
## Usage
### 1. Train the Model
Run the train.py script to perform training and validation. This script includes data augmentation, logging, confusion matrix & classification report generation, as well as attention weight statistics and visualization.

<br>``` python train.py ```

<br> During training, the best model weights will be automatically saved as best_model_attention.pth. Additional files generated include:
<br>* training_log.txt: Logs for each epoch.

<br>* training_curves.png: Plots of training and validation loss & accuracy.
![training_curves](https://github.com/user-attachments/assets/c7f6ebdf-b9dd-45b7-9f28-2462e7d68bd3)

<br>* confusion_matrix.txt and classification_report.txt: Evaluation metrics.
<br>* attention_weights.xlsx: Excel file with attention weight statistics, along with visualizations (boxplot and bar chart images).

![attention_weights_barplot](https://github.com/user-attachments/assets/ea4ed0a7-9ee8-4961-8094-17727901943c)


### 2. Evaluate and Visualize with Grad-CAM
<br>Run the eval_gradcam.py script to evaluate the model on the validation set and generate Grad-CAM visualizations.

<br>    ```python eval_gradcam.py```

<br>This script will:
<br>* Load the best saved model weights.

<br>* Predict on the validation set and save correctly and incorrectly classified samples into the correct_samples and incorrect_samples directories.
![图片1](https://github.com/user-attachments/assets/e0746588-7cd5-4f43-a464-3eed399bd871)


<br>* Generate Grad-CAM visualizations (original images, heatmaps, and overlay images) for each view (side, back, belly) and save them in the Grad-CAM directory.
![图片2](https://github.com/user-attachments/assets/ffa5722b-91ba-4656-bed6-f31bdb01a827)

## Model Architecture
<br>* Multi-branch ResNet: Three independent ResNet18 branches extract features from the side, back, and belly views.
<br>* Attention Mechanism: An attention module fuses the features from all branches by learning the relative importance of each view.
<br>* Classifier: A fully connected layer is used to classify the fused features into fish species.
## Important Notes
<br>* Data Path: Ensure your dataset directory and structure match those expected by the code to avoid file loading errors.
<br>* Pretrained Models: The code utilizes pretrained ResNet18 models (IMAGENET1K_V1). Make sure your environment can download these or they are cached.
<br>* Multi-GPU Support: The code supports multi-GPU training using nn.DataParallel.
<br>* Extensibility: Portions of the code have been reserved for potential future expansions, such as enhanced training functionalities.
## Contributing and Feedback
Feel free to submit issues or pull requests with suggestions for improvements. For any questions or feedback, please contact the project maintainer.

## License
Choose an appropriate open-source license for your project (e.g., MIT License) and include the details here.
