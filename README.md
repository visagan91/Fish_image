Multi-class image classification:Fish images
Introduction:
The aim of this project is to train multiple models and evaluate them and  deploy the best model in a streamline model to solve a multi-class image classification problem from the given data.
Data analysis:
The given data consist of already spliced class of images. They are split into train, val, and test folders. 
Keras found 6225 images in train and 1092 images in validation across 11 classes.
System design:
Data preprocessing: Upon analysis the images are originally in the pixel bandwidth of 0-255. We rescale the images to model specific requirements in the scale of [0,1] to stabilise training. 
For the train set small random transforms are carried so that the model identifies natural variations like rotation, shift, zoom etc.,  
CNN ‘from-scratch’: A compact CNN with 4 convolutional blocks of Conv-> ReLU-> MaxPool is created for a getting a baseline accuracy from the set of images.
Transfer learning: Pre-trained models on ImageNet is adapted to the set of images. The models used are VGG16, ResNet50, MobileNet, InceptionV3 and EfficientNetB0. 
Model Training: Training is done in two stages. One with a frozen base layer and another fine-tune stage where 20% of the top layers are unfreeze roughly. 
Few checkpoints are initialised for better handling and future evaluation.
Evaluation: Efficient evaluation metrics for each epoch is call-back for evaluating the models. The accuracy and loss per epoch is compared and validated for overfitting and fine-tuning further.
A confusion matrix is plotted to compare the accuracy across the models
Model selection: The transfer learned models are saved in .h5 format and a separate script loads each model and evaluate to choose the model with highest val accuracy before saving it .keras format. 
Streamlit app: A interactive interface is provided for the user to pick an image and predict the class. The best model selected is loaded and used to predict and display the results.
Challenges:
Since the process is complex the CPU performance was time consuming and over weighing. To over come this Apple Metal M1 GPU acceleration plugin from TensorFlow was used for training and inference.
EfficientNet pre-trained model was throwing error on image parameters. Upon learning that the model needs the image to be passed in RGB channels, ‘rgb’ inputs at 224x224 was used which fixed the weight-shaped error.
Results and inference:
The trained models are stored in a folder as follows
best_model.keras (Final model with highest val accuracy)
best_model_meta.json (the backbone related to the model won+val accuracy)
class-indices.jason (label mapping in the models)
The scratch model started with a low-level accuracy of ~0.17 and claimed fast and arched ~0.95-0.97 val accuracy.
The best transfer model was highly accurate with ~0.99 val accuracy making it the optimal choice.
There is a speed and size trade-off with accuracy among the models.
MobileNet: smallest + fastest; slightly lower accuracy
ResNet50 / InceptionV3 / EfficientNetB0: best accuracy; larger models.


