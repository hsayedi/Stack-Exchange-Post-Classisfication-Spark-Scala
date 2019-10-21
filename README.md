# Stack Exchange Post Classification

This assignment is on Stack Exchange post classification using the following four algorithms: Logistic Regression, Decision 
Tree, Naive Bayes, and Artificial Neural Network (Perceptron). 
There are two datasets: a seed dataset and a test set. The seed data is split into 70% training and 30% validation set,
while training. The test set (input_data.csv) is unlabeled and is the test set where we predict the post category, which 
consists of 10 distinct categories (labels).

Brief background on the algorithms used:
#### Logistic Regression:
A supervised classification model used when target variable is categorical. In our case, we use a multinomial logistic 
regression model where the number of classes are greater than 2. The model uses a elastic net which will run the binary
model across all classes then output the class with the highest score.
#### Decision Tree: 
Flowchart structure where each node represents a “test” on an attribute, each branch is the is the outcome of that test, and
each leaf node represents a class label (the decision or outcome of the test). The depth of the tree indicates how many 
nodes deep the tree. 
#### Naive Bayes:
Used for classification tasks. A way where probability theory could be used to classify things. 
Choose decision with the highest probability and then classify into classes based on that. The algorithm is based on 
Bayes theorem - which uses priors (prior knowledge) and connects conditional and marginal probabilities to each other.
#### ANN (Perceptron):
Systems that intend to replicate the way humans learn. The algorithm consists of input and ouput layers. Hidden layers 
transform the input into something the output can use. 
ANNs are made up of 3 essential components: units/neurons, connections/weights/parameters, biases. 
Activation functions like sigmoid, tanh, reLU are commonly used. Perceptrons consist of a single neuron network of 2 layers, 
also called feed-forward network.


## Getting Started

If using a Mac, the easiest way to install the proper technologies is to use Homebrew

#### 1. Install Homebrew
In your terminal, run: 

```/usr/bin/ruby -e “$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)”```

#### 2. Install xcode-select 
Installing xcode-select allows us to install Java, Scala, and Apache Spark. In terminal, run:

```xcode-select-install```

#### 3. Install Java 1.8
In terminal, run:

```brew cask install java```

#### 4. Install Scala
In terminal, run:

```brew install scala```

#### 5. Install Spark
In terminal, run:

```brew install apache-spark```

You can verify if Spark installed properly by running the following in your Terminal: 

```spark-shell```
You should see the following if installation was successful: 


![GitHub Logo](/images/spark-shell.png)



## Running the code

To run the code, you can use spark-submit shell script that manages Spark applications.
In the root directory of the project, run in Terminal: 

```spark-submit --class SparkApp target/scala-2.12/spark-ml-examples_2.12-0.0.1-SNAPSHOT.jar```


## Outputs 
Below are some outputs after model testing. 

#### Logistic Regression:

![GitHub Logo](/images/LR.png)

#### Decision Tree:

![GitHub Logo](/images/DT.png)

#### Naive Bayes:

![GitHub Logo](/images/NB.png)

#### ANN (Perceptron):

![GitHub Logo](/images/ANN.png)



## Tech/Framework Used
Language: Scala

Framework: Spark 2.4.4



