# ERsProject

This project analyzes uploaded group images and returns detected people, estimated age, gender, emotion, approximate social roles, relationship labels, and an annotated image.

## Important Dataset Statement

This project does not train a custom machine learning model. It uses pretrained models through the `deepface` Python library and OpenCV.

For project/report submission, the safest wording is:

> No custom training dataset was used in this project. The system uses pretrained DeepFace models for age, gender, emotion, and RetinaFace-based face detection. OpenCV pretrained Haar cascade XML files are used for smile-related detection support. The dataset folder, if submitted, contains only sample test/input images and public reference dataset links.

Do not write that this project trained on FER2013, UTKFace, Adience, or any other dataset unless you actually trained a model with that dataset.

## Models Used

- Face detection: `RetinaFace` through `DeepFace.analyze(..., detector_backend="retinaface")`
- Age estimation: DeepFace pretrained age model
- Gender classification: DeepFace pretrained gender model
- Emotion recognition: DeepFace pretrained emotion model
- Smile detection support: OpenCV Haar cascade `haarcascade_smile.xml`
- Relationship prediction: rule-based logic in the project code, not a trained ML model

## DeepFace Links

- DeepFace GitHub repository: https://github.com/serengil/deepface
- DeepFace pretrained model weights repository: https://github.com/serengil/deepface_models
- DeepFace pretrained model releases: https://github.com/serengil/deepface_models/releases/
- DeepFace demographic model reference: https://deepwiki.com/serengil/deepface/5.2-demographic-models

DeepFace downloads pretrained weights automatically when required. They are normally cached in the user's home directory under:

```text
~/.deepface/weights/
```

Relevant pretrained weight files include:

- `age_model_weights.h5`
- `gender_model_weights.h5`
- `facial_expression_model_weights.h5`
- `retinaface.h5`

## Reference Dataset Links

These are reference datasets related to the model tasks. They are not custom training datasets for this project unless you separately download and train with them.

### Emotion Recognition

- FER2013 facial expression dataset:
  https://www.kaggle.com/datasets/pankaj4321/fer-2013-facial-expression-dataset

Typical folder structure:

```text
FER2013/
+-- train/
+-- val/
+-- test/
```

Emotion classes usually include:

```text
angry, disgust, fear, happy, neutral, sad, surprise
```

### Age and Gender Estimation

- UTKFace official dataset page:
  https://susanqq.github.io/UTKFace/
- UTKFace Kaggle mirror:
  https://www.kaggle.com/datasets/jangedoo/utkface-new

UTKFace image filenames usually encode labels in this format:

```text
age_gender_race_date.jpg
```

Example:

```text
25_0_2_201701161745.jpg
```

### Age and Gender Benchmark Reference

- Adience benchmark information:
  https://exposing.ai/adience/

### OpenCV Haar Cascade Files

- OpenCV Haar cascades:
  https://github.com/opencv/opencv/tree/master/data/haarcascades

The project uses OpenCV's installed cascade files through:

```python
cv2.data.haarcascades
```

## Suggested Dataset Folder for Submission

If your college asks for a dataset folder, use a small and honest folder like this:

```text
datasets/
+-- sample_test_images/
|   +-- test.jpg
|   +-- group_sample_1.jpg
+-- README.txt
```

The `datasets/README.txt` can say:

```text
This folder contains sample input images used to test the application.
The project does not include a custom training dataset because it uses pretrained DeepFace and OpenCV models.
Public reference datasets are listed in the main README.md file.
```

## Kaggle Download Commands

Only use these if you specifically need to download public reference datasets:

```bash
pip install kaggle
kaggle datasets download -d pankaj4321/fer-2013-facial-expression-dataset -p datasets/FER2013 --unzip
kaggle datasets download -d jangedoo/utkface-new -p datasets/UTKFace --unzip
```

Kaggle downloads require a Kaggle account and API token.

## Run Instructions

Backend:

```bash
cd C:\Users\Ritwik\ERsProject\backend
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install flask flask-cors opencv-python deepface numpy tf-keras
python app.py
```

Frontend:

```bash
cd C:\Users\Ritwik\ERsProject\frontend
npm install
npm start
```

Open the frontend at:

```text
http://localhost:3000
```
