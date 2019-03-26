# Brain Tumor Classification

Deep-Neural Network for Brain Tumor Classification in Pediatric Patients

## Tumor Classification Categories

-   `DIPG` - Diffuse Intrinsic Pontine Glioma
-   `EP` - Ependymomas
-   `MB` - Medulloblastoma
-   `PILO` - Pilocytic
-   `Normal` - Healthy Brain

## Packages and Versions

| **Package Name**      | **Version** |
| --------------------- | ----------- |
| `python`              | 3.6         |
| `tensorflow`          | 1.12        |
| `keras`               | 2.1.5       |
| `keras-applications`  | 1.0.7       |
| `keras-preprocessing` | 1.0.9       |
| `matplotlib`          | 3.0.2       |
| `imageio`             | 2.4.1       |
| `Pillow*`             | 5.4.1       |
| `scikit-learn`        | 0.20.2      |
| `scikit-image`        | 0.14.1      |
| `numpy`               | 1.15.4      |
| `pandas`              | 0.24.1      |
| `pydicom*`            | 1.2.2       |
| `nibabel`             | 2.3.3       |
| `progressbar`         | 2.5         |

`*` - See note below regarding OpenJPEG

## Files

## Folders

-   `source` - Folder for storing the raw DICOM/NII images
    -   `DIPG`
        -   `Seattle` - test images
        -   `Stanford`
    -   `PILO`
        -   `Stanford`
    -   `MB`
        -   `Seattle` - test images
        -   `Stanford`
    -   `EP`
        -   `Seattle` - test images
        -   `Stanford`
    -   `Normal` - Healthy Children Brain Scans
    -   `katie_annotated_metadata` - metadata for the Tumor dataset
    -   `Task01_Brain Tumor` - From the BRATS 2018 dataset
        -   `imagesTr` - Training images
        -   `imagesTs` - Testing images
        -   `labelTr` - Labels for Training images (For segmentation)(ignored)
        -   `dataset.json` - metadata for this dataset
    -   `flipped_clinical_NormalPedBrainAge_StanfordCohort.csv` - metadata for healthy brains
-   `docs` - Folder for storing static content and documents
-   `weights` - Folder to store all model weights
-   `models` - Folder to store all model codes
-   `logs` - Folder to store all logs

## [NOTE] JPEG2000 decoding

Pillow needs to be installed with support for JPEG2000 lossless compression, which is the compression used in some the MRI DICOM scans. In order to do that, install OpenJPEG before installing Pillow. The steps to install OpenJPEG are -
0.  Install cmake to build OpenJPEG (see [this](https://pachterlab.github.io/kallisto/local_build.html) for local user installation and avoiding sudo in step 7 and 8)
1.  `git clone https://github.com/uclouvain/openjpeg.git`
2.  `cd openjpeg`
3.  `mkdir build`
4.  `cd build`
5.  `cmake .. -DCMAKE_BUILD_TYPE=Release`
6.  `make`
7.  `sudo make install`
8.  `sudo make clean`

Restart the session.
Now install Pillow, followed by pydicom

## [NOTE] Keras Version issues

-   During this and many other projects, I noticed that there are many open issues regarding model checkpointing in Keras 2.2 onwards. For this reason, I have stuck to training the models on Keras 2.1.5.
-   One of the issues comes with saving the model (weights + architecture + optimizer state) with Lambda layers which have Keras/Tensorflow arguments as input.
    -   See this [issue](https://github.com/keras-team/keras/issues/8343) and [issue](https://github.com/keras-team/keras/issues/10528). They suggest changing a lot of things, amounting to hacking the system to work.
    -   This [answer on StackOverflow](https://stackoverflow.com/questions/47066635/checkpointing-keras-model-typeerror-cant-pickle-thread-lock-objects) gives a very detailed explanation of the issue. This basically involves implementing Tensor-to-numpy-to-Tensor transformations. These seem too expensive an operation for a task that is going to be executed millions of times during training.
-   If we avoid saving the model along with architecture and optimizer state and just save the weights, we can solve the above issue. But since Keras 2+, saving models with Siamese networks or Transfer learning layers tend to fail. It tends to throw `ValueError: axes don't match array` when we are trying to load the saved weights into the model again. This issue doesn't exist below 2.1.6 to my knowledge right now.
    -   See this [issue](https://github.com/keras-team/keras/issues/10428). [This](https://github.com/keras-team/keras/issues/10428#issuecomment-418303297) suggestion seems to work everytime. In this project, thankfully, Tensorflow 1.12 doesn't seem to cause any issue. Maybe someone should update the issue...
    -   Another [issue](https://github.com/experiencor/keras-yolo2/issues/358) where the same error arises. The same suggestion seems to work again!
