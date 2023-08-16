# Outer Space Prediction Model Documentation

## Documentation

### Model description
The Outer Space Prediction Model is a supervised machine learning model that takes the relative velocity and distance of a body mass in outer space and predicts whether it is hazardous or not based on a Logistic Regression algorithm.

### Usage

#### Installation
```bash
pip install scikit-learn numpy
```
Install dependencies in the requirements.txt file:
```bash
pip install -r requirements.txt
```
or install them one by one using pip install.
##### requirements.txt
```bash
streamlit==1.3.1
pandas==1.3.5
numpy==1.21.5
sklearn==0.0
scikit-learn==1.0.2
matplotlib==3.5.1
```
Importing the Model
```python
from sklearn.linear_model import LogisticRegression
```
Creating an Instance
```python
model = LogisticRegression()
```
Making Predictions
```python
relative_velocity = 13569.2492241812
distance = 54839744.082846
prediction = model.predict([[relative_velocity, distance]])
print('Result is:', prediction)
```
### Inputs and Outputs

#### Inputs

The model takes the following inputs:

- Relative Velocity: Velocity of mass body in kmph.
- Distance: Distance of mass body from Earth in km.

#### Outputs
The model produces the following output:

- Predicted Output: Whether the mass body is hazardous to Earth or not.

## Example
Example Code:
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

relative_velocity = 13569.2492241812
distance = 54839744.082846
prediction = model.predict([[relative_velocity, distance]])

print('Result is:', prediction)
```

### UI Usage

It is pretty simple. You have two options:
1 - Enter only one entry. Just type in the relative velocity and distance of the mass body.
2 - Upload a CSV file that includes both of these features titled relative_velocity and miss_distance.

Then, click Predict. And voila! You will get the output either in the form of a message if Option 1 was selected, or a CSV file if Option 2 was selected.

### Notes

- This example assumes you have a model class LogisticRegression with methods for prediction.
- Replace the example class and methods with your actual model implementation.
- Ensure you have the required dependencies installed.
- Always preprocess input data appropriately before using the model for prediction.

### Creators

Eyad Magdy Gaber Khedr Hussein   
Mahmoud Akram Mahmoud Mohamed   
Toqa Alaa Abdel Rasoul Awad   
Nada Ashraf Moussa Kamel   
Ziad Hesham Al-Safy Rohayiem   
Steven Adel Ata Yakoub   
