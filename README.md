# Brain Tumor Classification

Deep-Neural Network for Brain Tumor Classification in Pediatric Patients

## Tumor Classification Categories

-   `DIPG` - Diffuse Intrinsic Pontine Glioma
-   `EP` - Ependymomas
-   `MB` - Medulloblastoma
-   `PILO` - Pilocytic
-   `Normal` - Healthy Brain

## Packages, Versions and Installation instructions

| **Package Name**      | **Version** |
| --------------------- | ----------- |
| `python`              | 3.6         |
| `tensorflow`          | 1.12        |
| `keras`               | 2.1.5       |
| `keras-applications`  | 1.0.7       |
| `keras-preprocessing` | 1.0.9       |
| `matplotlib`          | 3.0.2       |
| `imageio`             | 2.4.1       |
| `Pillow` \*           | 5.4.1       |
| `scikit-learn`        | 0.20.2      |
| `scikit-image`        | 0.14.1      |
| `numpy`               | 1.15.4      |
| `pandas`              | 0.24.1      |
| `pydicom*`            | 1.2.2       |
| `nibabel`             | 2.3.3       |
| `progressbar`         | 2.5         |

`*` - See note below regarding OpenJPEG prerequisite

To install the above packages -

-   if `python` is symlink to `python 3.x` --> `pip install --user <Package Name>`
-   if `python` is symlink to `python 2.x` --> `pip3 install --user <Package Name>`

## Transfer Learning Layers

| **Transfer Learning Model** | **Model Key**     |
| --------------------------- | ----------------- |
| ResNet 50                   | `resnet`          |
| Inception V3                | `inception`       |
| InceptionResNet V2          | `inceptionresnet` |
| Xception                    | `xception`        |
| DenseNet 121                | `densenet`        |
| VGG 19                      | `vgg`             |

## Instructions to Train and Evaluate the Models

1.  Install the above packages and dependencies.

2.  `git clone https://github.com/aman-chauhan/brain-tumor-classification.git`

3.  Set up `source` directory as shown below in **_Folder_** section.

4.  `python preprocessing.py` --> generates the `data` directory, which contains the cleaned, resized and partitioned data to train and evaluate the models. Also generates the `meta` directory, which contains the list of files required for reading the a particular partition.

5.  **Train the AutoEncoders** - There are 6 types of models, each using specific kinds of Transfer Learning layers. Refer **Model Key** from the **_Transfer Learning Layers_** section. The models are configured with EarlyStopping and can continue training over multiple runs. Train all these models. The command to train the models is --> `python train.py autoencoder <Model Key> <Batch Size>`. Examples -
    -   `python train.py autoencoder vgg 48`
    -   `python train.py autoencoder densenet 48`


6.  **Train the Classifiers** - Train the Classifiers associated with each type of Model (Model Key). The code also tunes the **dropout** hyper-parameter for the Fully Connected Layers between 0.1 and 0.5 inclusive. The command to train the models is --> `python train.py classifier <Model Key> <Batch Size>`. Examples -
    -   `python train.py classifier vgg 48`
    -   `python train.py classifier densenet 48`


7. `python vectorization.py` - Generate the vectors required for training the Paraclassifiers. Also identifies and stores the best hyperparameters for each type of Model. This will create the `paraclassifier` subdirectory inside `data` directory and `para_*.csv` inside `meta` directory.

8. **Train the Paraclassifiers** - Train the Classifiers associated with each type of Model (Model Key). This code can also train an ensemble model using the average vectors from all the models. The command to train the models is --> `python train.py paraclassifier <Model Key> <Batch Size>`. Examples -
    -   `python train.py paraclassifier vgg 32`
    -   `python train.py paraclassifier densenet 32`
    -   `python train.py paraclassifier ensemble 32`


9.  **Evaluate the Classifiers** - Evaluate the accuracy of the classifiers on test dataset. `python test.py classifier` --> This generates the `clf_results.csv` table in `logs` directory.

10.  **Evaluate the Paraclassifiers** - Evaluate the accuracy of the paraclassifiers on test dataset. `python test.py paraclassifier` --> This generates the `para_results.csv` table in `logs` directory. This is the final accuracy of our models.

## Instructions to Run Inference using our Models

1.  Switch to root of repository.
2.  `jupyter notebook` --> Start the Jupyter Notebook environment.
3.  Open `Tumor Classifier.ipynb`.
4.  Put in the path to your MRI scan in the **path** variable.
5.  Execute all the cells in the Notebook.

## Notebooks

-   `Tumor Classifier.ipynb` - Notebook for using our model to predict class of tumor, ie Inference using our Model.
-   `Exploring Data.ipynb` - Notebook for visualizing the different types of MRI scans present in the Data set.
-   `Visualization - AutoEncoder.ipynb` - Notebook for visualizing the results from training the AutoEncoder.
-   `Visualization - Classifier.ipynb` - Notebook for visualizing the results from training the Classifier.
-   `Visualization - Paraclassifier.ipynb` - Notebook for visualizing the results from training the Paraclassifier.

## Files

-   `preprocessing.py` - Code to clean data and preprocess.
-   `vectorization.py` - Code to generate vectors for each plane in brain.
-   `generator.py` - Code for Generator classes to train the models.
-   `train.py` - Code for training the Models using training and validation datasets.
-   `test.py` - Code for evaluating the Models against the test dataset, and generating the overall statistics.

## Folders

-   `source` - Folder for storing the raw DICOM/NII images
    -   `DIPG`
        -   `Seattle`
        -   `Stanford`
    -   `PILO`
        -   `Stanford`
    -   `MB`
        -   `Seattle`
        -   `Stanford`
    -   `EP`
        -   `Seattle`
        -   `Stanford`
    -   `katie_annotated_metadata` - metadata for the Tumor dataset
    -   `Normal` - Healthy Children Brain Scans
    -   `flipped_clinical_NormalPedBrainAge_StanfordCohort.csv` - metadata for healthy brains
    -   `Task01_Brain Tumor` - From the BRATS 2018 dataset. Download from [here](https://drive.google.com/uc?export=download&id=1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU)
        -   `imagesTr` - Training images
        -   `imagesTs` - Testing images
        -   `labelTr` - Labels for Training images (For segmentation)(ignored)
        -   `dataset.json` - metadata for this dataset


-   `data` - Folder to store cleaned data (generated by `preprocessing.py`)
    -   `autoencode` - Folder for cleaned files for AutoEncoder
        -   `train` - Folder for cleaned train files
        -   `valid` - Folder for cleaned validation files
    -   `classifier` - Folder for cleaned files for Classifier
        -   `train` - Folder for cleaned train files
        -   `valid` - Folder for cleaned validation files
        -   `test` - Folder for cleaned test files
    -   `paraclassifier` - Folder for cleaned files for Paraclassifier
        -   `train` - Folder for cleaned train files
        -   `valid` - Folder for cleaned validation files
        -   `test` - Folder for cleaned test files


-   `docs` - Folder for storing static content and documents
    -   `autoencoder` - Images and Graphs related to Autoencoder Visualizations
    -   `classifier` - Images and Graphs related to Classifier Visualizations
    -   `paraclassifier` - Images and Graphs related to Paraclassifier Visualizations
    -   `reference` - Reference implementation plots of the Model in Object Oriented Fashion
    -   `densenet` - Reference implementation plots of the Model in with DenseNet 121
    -   `inception` - Reference implementation plots of the Model in with Inception V3
    -   `inceptionresnet` - Reference implementation plots of the Model in with InceptionResNet V2
    -   `resnet` - Reference implementation plots of the Model in with ResNet 50
    -   `vgg` - Reference implementation plots of the Model in with VGG 19
    -   `xception` - Reference implementation plots of the Model in with Xception


-   `weights` - Folder to store all model weights
-   `models` - Folder to store all model codes
-   `logs` - Folder to store all logs
-   `meta` - Folder to store all the training metadata

## [NOTE] JPEG2000 decoding

Pillow needs to be installed with support for JPEG2000 lossless compression, which is the compression used in some the MRI DICOM scans. In order to do that, install OpenJPEG before installing Pillow. The steps to install OpenJPEG are -
1.  Install cmake to build OpenJPEG --> see [this](https://pachterlab.github.io/kallisto/local_build.html) for local user installation and avoiding sudo in step 8 and 9
2.  `git clone https://github.com/uclouvain/openjpeg.git`
3.  `cd openjpeg`
4.  `mkdir build`
5.  `cd build`
6.  `cmake .. -DCMAKE_BUILD_TYPE=Release`
7.  `make`
8.  `sudo make install`
9.  `sudo make clean`

Restart the session.
Now install Pillow, followed by pydicom
