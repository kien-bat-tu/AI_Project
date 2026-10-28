# MRL Infrared Eye Images Dataset for Drowsiness Detection (Forked Version)

This dataset is a **forked version** of the original MRL Eye Dataset, containing infrared eye images categorized into **Awake** and **Sleepy** states. It is split into training, validation, and test sets, comprising 84,898 images captured under various lighting conditions using multiple sensors. This dataset is tailored for tasks such as eye detection, gaze estimation, blink detection, and drowsiness analysis in computer vision.

## Dataset Structure:
- **Train**: Awake (25,770), Sleepy (25,167)
- **Validation**: Awake (8,591), Sleepy (8,389)
- **Test**: Awake (8,591), Sleepy (8,390)

## Directory Tree:
```data/ train/ awake/ sleepy/ val/ awake/ sleepy/ test/ awake/ sleepy/ ```


## Metadata:
- **subject_id**: Unique identifier for each subject (37 subjects)
- **Attributes**: Eye state, gender, glasses, reflections, lighting, and sensor ID

## Original Dataset Overview

The **MRL Eye Dataset** is a large-scale dataset of human eye images designed for computer vision tasks such as eye detection and blink detection. The original dataset includes 84,898 images captured under various conditions.

## Downloads:
- **Forked Dataset**: [Download Forked Dataset](http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip)
- **Original Dataset**: [Download Original Dataset](http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip)
- **Pupil Annotations**: [Download Pupil Annotations](http://mrl.cs.vsb.cz/data/eyedataset/pupil.txt)

## Contact:
For any questions about the dataset, please contact [Radovan Fusek](http://mrl.cs.vsb.cz//people/fusek/).

## Example Images

The dataset includes both open and closed eye images. Below are examples:

![Eye Image Example](http://mrl.cs.vsb.cz/images/eyedataset/eyedataset01.png)

## References


## Train the Classifier

The original dataset folders remain untouched. During training, the script
combines their file paths in memory and creates deterministic, subject-disjoint
train/validation/test groups from the subject ID in each filename. This avoids
the same person's eyes appearing in both training and evaluation. From the
project root, check the image counts and expected folder structure with:

```powershell
$python = 'C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe'
& $python data\split_data.py
```

On this Windows machine, use the selected Python interpreter to install the
project dependencies and check the dataset/model without starting training:

```powershell
$python = 'C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe'
& $python -m pip install -r requirements.txt
& $python data\train_model.py --smoke-test
```

Train the CNN. The best checkpoint is selected by subject-disjoint validation
accuracy and saved outside the dataset at `models/eye_state_cnn.pt`; the
subject-disjoint test set is evaluated only after training finishes. A short
CPU-friendly run is:

```powershell
$python = 'C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe'
& $python data\train_model.py --epochs 5 --batch-size 128
```

To classify one eye image using the saved model:

```powershell
$python = 'C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe'
& $python data\train_model.py --predict 'path\to\eye.png'
```

Or open a file picker to select an image:

```powershell
$python = 'C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe'
& $python data\train_model.py --choose-image
```

The simplest way to start the demo is to double-click `open_eye_demo.bat` in
the project folder. In the app window, click **CHỌN ẢNH**, select an image, and
click Open. The selected image is shown with separate predictions for the
left and right eye as viewed in the photo; choose another image to try again.

For a face photo, the demo first uses OpenCV Haar detection to find the largest
face, estimates separate left/right eye regions, classifies each region, and
draws a colored ellipse around each estimate. Green means `awake` (open); red
means `sleepy` (closed/sleepy). Each eye's prediction and validation-calibrated
confidence percentage appear on separate lines. The percentage is calibrated
on infrared validation images, not selfie photos, so it is only an estimate
for phone images. If no
face is detected, the demo asks for a clearer, front-facing photo instead of
classifying the whole face as if it were an eye.

Haar detection is a lightweight classical detector, so profile views, strong
occlusion, or poor lighting may prevent face detection. The CNN was trained on
infrared eye crops, not color selfies, so its confidence on phone photos is an
estimate and may be unreliable. This demo processes still images only.

