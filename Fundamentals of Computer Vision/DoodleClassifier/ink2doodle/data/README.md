# Downloading the Data

For this project, we will be using the Quick, Draw! Dataset. The dataset is available on Google Cloud Storage. You can download the data fby running the following commands in the terminal:

```bash
gsutil -m cp 'gs://quickdraw_dataset/full/numpy_bitmap/*' .
```

You will need to install the `gsutil` package to run the above command. You can install it by following the instructions [here](https://cloud.google.com/storage/docs/gsutil_install).

# Training the model

We also recomand to donload the class names file with the following command if you plan to train the model:

```bash
wget 'https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt' -o categories.txt
```
