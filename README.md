### ABOUT
This is the source code of the online stage of DeepPower, and the source code of the training process in the offline stage is temporarily reserved.

The source code contains two steps in the online phase, and the trained weight files are saved in the "results" folder under the "Step1_Network_Structure_Recovery" and "Step2_Layer-wise_Hyperparameter_Inferring" folders.

In addition, a small-scale energy dataset is provided for evaluation and testing, which contains energy traces of nine standard architectures, including ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, VGG11, VGG13, VGG16, and VGG19.

### DEPENDENCIES
Our code is implemented and tested on PyTorch. Following packages are used by our code.
- `tensorflow-gpu==110.1`
- `torchvision==0.11.2`
- `numpy==1.22.0`
- `python-Levenshtein==0.12.2`

### RUN
Evaluate online phase step 1.
```python
python Step1_Network_Structure_Recovery/eval.py
```
Evaluate online phase step 2.
```python
python Step2_Layer-wise_Hyperparameter_Inferring/eval.py
```
