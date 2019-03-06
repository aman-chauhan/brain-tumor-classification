# Brain Tumor Classification

Deep-Neural Network for Brain Tumor Classification in Pediatric Patients

## Tumor Categories

-   `DIPG` - Diffuse Intrinsic Pontine Glioma
-   `EP` - Ependymomas
-   `MB` - Medulloblastoma
-   `PILO` - Pilocytic
-   `Pediatric`

## Packages and Versions

-   `python` - version 3.6
-   `tensorflow` - version 1.12
-   `keras` - version 2.1.5
-   `keras-applications` - version 1.0.7

## Files

## Folders

-   `data` - Folder for storing the raw DICOM images
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
-   `docs` - Folder for storing static content and documents
-   `weights` - Folder to store all model weights
-   `models` - Folder to store all model codes
-   `logs` - Folder to store all logs

## [NOTE] Keras Version issues

-   During this and many other projects, I noticed that there are many open issues regarding model checkpointing in Keras 2.2 onwards. For this reason, I have stuck to training the models on Keras 2.1.5.
-   One of the issues comes with saving the model (weights + architecture + optimizer state) with Lambda layers which have Keras/Tensorflow arguments as input.
    -   See this [issue](https://github.com/keras-team/keras/issues/8343) and [issue](https://github.com/keras-team/keras/issues/10528). They suggest changing a lot of things, amounting to hacking the system to work.
    -   This [answer on StackOverflow](https://stackoverflow.com/questions/47066635/checkpointing-keras-model-typeerror-cant-pickle-thread-lock-objects) gives a very detailed explanation of the issue. This basically involves implementing Tensor-to-numpy-to-Tensor transformations. These seem too expensive an operation for a task that is going to be executed millions of times during training.
-   If we avoid saving the model along with architecture and optimizer state and just save the weights, we can solve the above issue. But since Keras 2+, saving models with Siamese networks or Transfer learning layers tend to fail. It tends to throw `ValueError: axes don't match array` when we are trying to load the saved weights into the model again. This issue doesn't exist below 2.1.6 to my knowledge right now.
    -   See this [issue](https://github.com/keras-team/keras/issues/10428). [This](https://github.com/keras-team/keras/issues/10428#issuecomment-418303297) suggestion seems to work everytime. In this project, thankfully, Tensorflow 1.12 doesn't seem to cause any issue. Maybe someone should update the issue...
    -   Another [issue](https://github.com/experiencor/keras-yolo2/issues/358) where the same error arises. The same suggestion seems to work again!
