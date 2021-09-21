# Deployment

Let's say I have developed a model, and would like to deploy it so the model can be used by other people. 


```python
# Standard Imports
import pandas as pd
import matplotlib.pyplot as plt

# Model Development
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector

# Stock Data
from sklearn.datasets import load_iris
```


```python
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df.drop('target', axis=1)
y = df.target
```


```python
model = LogisticRegression(penalty='none')

sfs = SequentialFeatureSelector(model, n_features_to_select=2)
sfs.fit(X, y)
```




    SequentialFeatureSelector(estimator=LogisticRegression(penalty='none'),
                              n_features_to_select=2)




```python
sfs.get_support()
```




    array([False,  True, False,  True])




```python
X.columns[sfs.get_support()]
```




    Index(['sepal width (cm)', 'petal width (cm)'], dtype='object')




```python
sfs.transform(X)[:5]
```




    array([[3.5, 0.2],
           [3. , 0.2],
           [3.2, 0.2],
           [3.1, 0.2],
           [3.6, 0.2]])




```python
X_subset = sfs.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_subset, y)

model.fit(X_train, y_train)

fig, axes = plt.subplots(1,2, figsize=(20,6))

plot_confusion_matrix(model, X_train, y_train, ax=axes[0])
plot_confusion_matrix(model, X_test, y_test, ax=axes[1])

axes[0].set_title('Training', fontsize=20)
axes[1].set_title('Testing', fontsize=20);
```


![png](README_files/README_9_0.png)


### Fit model to all available data


```python
final_model = model.fit(X_subset, y)
```

### Pickle final model


```python
import pickle

file = open('final_model.pkl', 'wb')
pickle.dump(final_model, file)
file.close()
```

### Now what?

# Deployment Patterns

> The following is a breakdown of existing deployment patterns, via Data Scientist Andrew Ng's MlOps MOOC

**Some common themes:**
- Deployments are usually ramped up over time.
- Monitoring performance is key part of production level deployment.
- Good deployment strategies include the ability to rollback to a previous model.


### Shadow Deployment

- Model shadows a human worker and generates predictions in parallel
- The outcome does not utilize the model's predictions.
- This deployment pattern serves a further evaluation of the model by collecting new labels on out of sample data and comparing them with your model's predictions. 

###  Canary Deployment

- The model is applied to a small percent of traffic.
- The model is monitored via specified metrics and is ramped up over time. 


### Blue Green Deploymenr

- `Blue` represents a currently deployed model that you would like to replace
- `Green` represents a new model
- Both `Blue` and `Green` are running on their own webservice/api.
- Using this deployment pattern allows you to easily reroute the data from one model to another. When something goes wrong with a model, it is very easy to default back to a simpler model with more reliable predictions. 

## Degrees of Automation

### Human only

No model is used. All decisions are manually generated via human workers. 

### Shadow Mode

The model is run in the background and does not impact the outcome.

### AI Assistance

The model does not make the final decision, but provides assistance to a human worker who makes the decision. Examples could be 
- Providing their prediction to the worker (Autocomplete).
- Highlighting parts of image that are likely relevant to a decision.
- Suggesting relevant resources to a worker. 

### Partial Deployment

The model's predictions are used to make a decision only when the model is a certain degree of confidence. When the model's confidence is low, the decision is given to a human worker. 


### Full Automation

There is no human worker involved in the decision process.

## Metrics

Model analysis does not end once the model has been deployed. A full deployment usually involes setting up a data collection pipeline to monitor inputs, outputs, and activity of your model.

These metrics tend to be highly customized for whatever the data that being inputted and the problem the model is addressing. 

**Example metrics:**

- Server load
- Descriptive statistics about the inputs and outputs
    - How do the distributions of the inputs features compare to the distributions the model was trained on?
    - Brightness of the images
    - Magnitudes of numerical inputs
    - Frequency of categoricals
    - How frequently does your model produce a confidence result?
    - Does the model's output distribution reflect the expected distribution?
- User results
    - Did they add the item to their cart
    - Did the use the word recommended to them?
    - Did they use the product from beginning to end or did the leave the website prematurely
    - Did the default to typing their question instead of your model interpreting their speech?
    
 

#### Intro to Flask

Flask ([Documentation](https://flask.palletsprojects.com/en/2.0.x/)) is a light weight python framework for developing apis and even [full blown websites](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world). It is a very popular framework used in the data science community because of it's simplicity. 





```python
from flask import Flask

app = Flask(__name__)

# This is
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/<name>")
def hello_name(name):
    return f"<p>Hello, {name}!</p>"

from waitress import serve
serve(app, host='0.0.0.0', port=5000)
```

    Serving on http://0.0.0.0:5000


### Deployment Template

We have provided a [deployment template repository](https://github.com/learn-co-students/flask-model-deployment), that allows you to quickly produce a simple webform so people can interact with your model. 

