# ML-Intro
This Github repo serves as an introduction to Machine Learning where I cover the basics of supervised and unsupervised learning  and the different algorithms that exist using Python and scikit-learn library.

![image](https://user-images.githubusercontent.com/51273123/234427227-1b2b7c12-e166-47ff-a5bb-2b9fe6d40031.png)


## Book recommendation

![Book Cover](mlbook.jpg)

I highgly recommend  [Keras to Kubernetes: The Journey of a Machine Learning Model to Production]([https://www.example.com/](https://www.amazon.com/Keras-Kubernetes-Journey-Learning-Production/dp/1119564832)) for anyone who wants to learn how to deploy machine learning algorithms in production, The book covers a variety of practical topics such as data preprocessing, model training, evaluation, and deployment.

## What is Machine Learning? 

Machine Learnaing is a subfield of **AI**, that involves building _models_ that can extract the value from the data by making predictions and informed decisions. They are different approaches to machine leanring, the most common are supervised, unsupervised and reinforcement learning.
For the sake of this article I will discuss only **Supervised** and **Unsupervised models**.

### Supervised Learning
This type of machine learning is defined by using labeled data (target variable )to train the model.The goal of supervised learning is to learn a mapping between input variables (features) and output variables (labels) based on the training data, so that the algorithm can predict the output value for new, unseen input data.

![image](https://user-images.githubusercontent.com/51273123/234430541-7bee2e30-319d-4dab-b421-01e2c32a0f64.png)

Supervised learning can be used for both regression and classification tasks. In regression tasks, the goal is to predict a continuous numerical value, such as the price of a house based on its features. In classification tasks, the goal is to predict a discrete class label, such as whether an email is spam or not based on its content.

![image](https://user-images.githubusercontent.com/51273123/234430974-73fa9dbb-a950-4760-b263-83b677aba762.png)

#### Regression
We predict a **continuous numerical** value as the _output_ based on the _input_ features. It involves finding the **relationship** between the **_independent variables_** (input features) and **_dependent variable_** (output) by fitting a _function_ to the training data. The fitted function can then be used to make predictions on new data points. The most common regression algorithms are linear and polynomial regression, decision tree, random forrest, support vector machine (SVM), nerual networks. 
#### Classification
In this type of machine learning the goal is to predict a **categorical** variable or label based on the input features. The input features can be continuous or discrete, and the output variable is a **_class or category_**. The classification can be binary (two classes) or multi-class (more than two classes). There are several algorithms used for classification tasks such as logistic regression, decision trees, random forests, support vector machines (SVMs), and neural networks, among others. The choice of algorithm depends on the specific task, the characteristics of the dataset, and the performance metrics used to evaluate the model.

----------------

### The 4 steps to create an efficient Machine Learning model

#### 1st step: The Dataset 

The first step in creating an efficient machine learning model is to preprocess the data. This involves cleaning the data, dealing with missing values, and handling outliers. It also involves converting the data into a suitable format for the model to work with, such as converting categorical variables into numerical values. This step is critical because the quality of the data has a significant impact on the accuracy of the model.

#### 2nd Step: The Model selection

The third step is to select an appropriate algorithm for the problem at hand and train the model on the data. The choice of algorithm depends on several factors such as the type of problem (classification, regression, etc.), the size and complexity of the dataset, and the performance metrics used to evaluate the model. It's important to split the data into training and testing sets to evaluate the performance of the model on unseen data.

#### 3rd Step: Cost Function 

In machine learning, the cost function is a measure of how well a machine learning model is performing on a given dataset. The cost function calculates the difference between the predicted output and the actual output for a given set of input features. The goal of the model is to minimize the cost function, i.e., to reduce the difference between the predicted output and the actual output as much as possible. 

#### 4th Step: Optimisation Algorithm

The optimization algorithm iteratively adjusts the model's parameters to minimize the cost function. In each iteration, the algorithm computes the gradient of the cost function with respect to the model parameters and updates the parameters accordingly to move in the direction of steepest descent. The learning rate is another important hyperparameter that controls the size of the step taken in each iteration. 




