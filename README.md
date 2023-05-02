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

### The 4 steps to create an efficient Supervised Machine Learning model

#### 1st step: The Dataset 

The first step in creating an efficient machine learning model is to preprocess the data. This involves cleaning the data, dealing with missing values, and handling outliers. It also involves converting the data into a suitable format for the model to work with, such as converting categorical variables into numerical values. This step is critical because the quality of the data has a significant impact on the accuracy of the model.

#### 2nd Step: The Model selection

The third step is to select an appropriate algorithm for the problem at hand and train the model on the data. The choice of algorithm depends on several factors such as the type of problem (classification, regression, etc.), the size and complexity of the dataset, and the performance metrics used to evaluate the model. It's important to split the data into training and testing sets to evaluate the performance of the model on unseen data.

#### 3rd Step: Cost Function 

In machine learning, the cost function is a measure of how well a machine learning model is performing on a given dataset. The cost function calculates the difference between the predicted output and the actual output for a given set of input features. The goal of the model is to minimize the cost function, i.e., to reduce the difference between the predicted output and the actual output as much as possible. 

#### 4th Step: Optimisation Algorithm

The optimization algorithm iteratively adjusts the model's parameters to minimize the cost function. In each iteration, the algorithm computes the gradient of the cost function with respect to the model parameters and updates the parameters accordingly to move in the direction of steepest descent. The learning rate is another important hyperparameter that controls the size of the step taken in each iteration. 

----------------------------------------

### The first Portal to Machine Learning models **Linear Regression**

Linear regression is a  statistical model that is simple to implement and understand, it's one of the first  algorithm that helped me understand the basic of machine learning.  Linear regression is a method of prediction in which the dependent variable (y) is modeled as a linear function of one or more independent variables (x1,x2,x3...). This algorithm has two types: **Simple Linear Regression** and **Multiple Linear Regression**. Simple Linear Regression uses only one independent (x) variable while Multiple Linear Regression uses multiple independent variables (x1,x2,x3...) like age, height etc.

----
1. [simple linear regression](https://github.com/Rezquellah/ML-Intro/blob/main/Simple_Linear_Regression.ipynb)

y = b + ax

where:

"y" is the dependent variable (or target variable)

"x" is the independent variable (also called feature)

"b" is the y-intercept (the value of y when x is 0)

"a" is the slope (the change in y for a one-unit increase in x)

This equation describes a straight line relationship between x and y. The goal of linear regression is to find the values of **"a"** and **"b"** that minimize the distance between the predicted values of y and the actual values of y in the data. This is typically done using a method called least squares regression.


2. [Multiple Linear Regression](https://github.com/Rezquellah/ML-Intro/blob/main/Polynomial_Regression.ipynb)

y = b0 + b1x1 + b2x2 + ... + bnxn + ε

Where y is the dependent variable, x1, x2, ..., xn are the independent variables, b0 is the intercept (the expected value of y when all independent variables are 0), b1, b2, ..., bn are the slopes (the change in y for a one-unit change in the corresponding independent variable), and ε is the error term (the difference between the predicted and actual values of y).

The equation can also be expressed in vector form as:

y = Xβ + ε

Where y is an n x 1 vector of the dependent variable, X is an n x (p+1) matrix of the independent variables (with the first column being all 1s for the intercept), β is a (p+1) x 1 vector of the coefficients (intercept and slopes), and ε is an n x 1 vector of the error terms.
-----

**Linear regression relies on several assumptions to be valid. The following are the key assumptions or rules of linear regression:**

Linearity: The relationship between the dependent variable and the independent variable(s) is linear.

No multicollinearity: The features shouldn't be correlated together

Normality: The datest need to follow a normal distribution 

No outliers: The data should not contain any extreme or influential observations that can distort the results.

Constant mean: The mean of the dependent variable is constant for all values of the independent variable(s).

------

**How to evaluate a linear regression model ?**

There are various methods for evaluating the performance of a linear regression model:

R-squared (R2):  this is the coefficient of determination and is a measure of how well your model fits the data. It represents the proportion of variance in your dependent variable (y) that can be explained by your independent variables (x). The closer this value is to 1, the better your linear regression model explains or predicts y.

Adjusted R-squared: Adjusted R-squared is a modified version of R-squared that penalizes the inclusion of additional independent variables that do not contribute much to the model. It is a better measure of model fit when there are multiple independent variables.

Cross-validation: Cross-validation is a technique for evaluating the performance of a model on new data. It involves dividing the data into training and testing sets and evaluating the model on the testing set. Cross-validation can help prevent overfitting and provide a more accurate estimate of the model's performance on new data.

Hypothesis testing: Hypothesis testing can be used to test the significance of the coefficients in the model. This can help determine which independent variables are important predictors of the dependent variable.

------------

**Limitations of Linear Regression**: Linear regression is a powerful statistical tool, but it has its limitations. It assumes a linear relationship between the independent variable(s) and the dependent variable, and it may not be appropriate for non-linear relationships. It also assumes that the data is independent, which may not be the case in some situations.

*Non-linear Relationships: If the data has a non-linear relationship, linear regression won't be able to capture it. For example, if we're trying to predict the price of gold based on its weight and purity (or vice versa), you'll need another method of analysis.

*Outliers: An outlier is an observation that falls outside of expected values or trends in your dataset, For example, if someone accidentally enters an incorrect value into their spreadsheet when calculating their age (e.g., typing "80" instead of "20"), this would be considered an outlier because it doesn't fit with what we know about human aging patterns across populations at large; however if we were looking at just one person's life span rather than many people's lifespans together then this would still be considered valid data even though it doesn't conform with general trends seen elsewhere within our sample size."

-------------

I have created a complete project that predict [fish weigh](https://github.com/Rezquellah/Fish_Weight_Prediction). The project includes exploratory data analysis, feature engineering, and model building and evaluation. I have also deployed the app in heroku at https://myfishweight.herokuapp.com/ as a container. 





