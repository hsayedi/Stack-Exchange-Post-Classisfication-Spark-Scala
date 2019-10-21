/*
Gemini Data Assignment
Husna Sayedi
 */

import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, MultilayerPerceptronClassifier, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IndexToString, RegexTokenizer, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

// Optional: Use the following code below to set/ the Error reporting
import org.apache.log4j._

/*
On terminal
sbt clean package

// Run using spark-submit
spark-submit --class SparkApp target/scala-2.12/spark-ml-examples_2.12-0.0.1-SNAPSHOT.jar

 */

object SparkApp {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR) // suppress some warnings

    // Spark Session
    val spark = SparkSession.builder().master("local[*]").getOrCreate()

    // Use Spark to read in the seed csv file
    val data = spark.read
      .option("header", "true") // shows headers
      .option("inferSchema", "true") // shows column types
      .option("quote", "\"") // escape double quotes
      .option("escape", "\"") // escape double quotes
      .option("delimiter", ",") // set delimiter as comma
      .option("multiLine", true)
      .option("ignoreTrailingWhiteSpace", true) // remove spaces
      .format("csv")
      .load("data/seed.csv")

    // CLEAN DATAAAAAAA

    // Print the Schema of the DataFrame
    data.printSchema()

    //  data.select("_Body", "_OwnerUserId", "_Title", "_Category").show()
    //  data.select("_Category").distinct.show()

    // Read the testing data
    val testData = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("quote", "\"")
      .option("escape", "\"")
      .option("delimiter", ",")
      .option("multiLine", true)
      .option("ignoreTrailingWhiteSpace", true)
      .format("csv")
      .load("data/input_data.csv")

    trainModels(data, testData)
  }

  def trainModels(data: DataFrame, testData: DataFrame): Unit = {

    println("Training Logistic Regression")
    val lr = new LogisticRegression() // which will run the binary version across all classes and return the class with the highest score.
      .setMaxIter(10) // set maximum iterations

    // Hyper parameter tuning
    // We use a ParamGridBuilder to construct a grid of parameters to search over for best model selection
    val paramGridLr = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01)) // lambda value
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0)) // where alpha is mixing parameter between ridge (alpha=0) and lasso (alpha=1)
      .build()

    mlTrainer(data, testData, lr, paramGridLr)


    println("Now training Decision Tree Classifier")
    val dt = new DecisionTreeClassifier()

    // Hyper parameter tuning
    // We use a ParamGridBuilder to construct a grid of parameters to search over for best model selection
    val paramGridDt = new ParamGridBuilder()
      .addGrid(dt.maxBins, Array(15, 20)) // bins = way of sorting data, putting data into categories / buckets
      .addGrid(dt.maxDepth, Array(2, 4)) // how many nodes deep our decision tree goes
      .build()

    mlTrainer(data, testData, dt, paramGridDt)


    println("Now training Naive Bayes Classifier")
    val nb = new NaiveBayes() // default is multinomial. NB is best for text classification problems

    // Hyper parameter tuning
    // We use a ParamGridBuilder to construct a grid of parameters to search over for best model selection
    val paramGridNb = new ParamGridBuilder()
      .addGrid(nb.smoothing, Array(0.0, 0.5, 1.0)) // lambda value - default is 1.0
      .build()

    mlTrainer(data, testData, nb, paramGridNb)


    println("Now training Artificial Neural Network (Perceptron)")
    val numOutputCategories = data.select("_Category").distinct().count().toInt // 10 distinct labels
    println(numOutputCategories)
    /* specific layers for ANN:
       num of features = 1000 since HashingTF produces 1000 features
       two intermediate layers of size 5 and 4
       the output is size of the distinct classes
     */
    val layers = Array[Int](1000, 5, 4, numOutputCategories)
    val nn = new MultilayerPerceptronClassifier()
      .setLayers(layers)

    // Hyper parameter tuning
    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    val paramGridNn = new ParamGridBuilder()
      .addGrid(nn.blockSize, Array(5, 128))
      .addGrid(nn.maxIter, Array(10, 100))
      .build()

    mlTrainer(data, testData, nn, paramGridNn)
  }

  def mlTrainer(dataRaw: DataFrame, testDataRaw: DataFrame, estimator: Estimator[_], paramGrid: Array[ParamMap]): Unit = {

    val train = dataRaw.withColumnRenamed("_Body", "body")
      .withColumnRenamed("_Category", "category")
      .select("body", "category")
    val testData = testDataRaw.withColumnRenamed("_Body", "body")
      .select("body")

    // Stages in the ML pipeline.
    val regexTokenizer = new RegexTokenizer() // first remove tags from string
      .setInputCol("body")
      .setOutputCol("removeTags")
      .setPattern("<[^>]+>")
//    val Tokenizer = new Tokenizer() // process of taking text and breaking into terms (words)
//      .setInputCol(regexTokenizer.getOutputCol)
//      .setOutputCol("words")
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(regexTokenizer.getOutputCol)
      .setOutputCol("removedStopWords")
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("features")
    val indexer = new StringIndexer() // create label - convert string to number
      .setInputCol("category")
      .setOutputCol("label")

    /* A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
       TrainValidationSplit used for hyper parameter tuning - evaluates each combination of parameters once as opposed
       to k times in the case of CrossValidator. It creates a single training/test dataset pair and splits between
       the training and testing is done based on trainRatio parameter
     */
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(estimator)
      .setEvaluator(new MulticlassClassificationEvaluator() // we have 10 distinct labels - so use Multiclass evaluator
        .setMetricName("accuracy"))
      .setEstimatorParamMaps(paramGrid) // using our parameter grids to find the optimal parameters
      // 70% of the data will be used for training and the remaining 30% for validation.
      .setTrainRatio(0.7)
      // Evaluate up to 2 parameter settings in parallel
      .setParallelism(2)


    // Get category back from label
    val labelReverse = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedCategory")
      .setLabels(indexer.fit(train).labels)

    // Set our ML pipeline
    val pipeline = new Pipeline()
      .setStages(Array(regexTokenizer, stopWordsRemover, hashingTF, indexer, trainValidationSplit, labelReverse))

    // Fit the train data
    val model: PipelineModel = pipeline.fit(train)

    // Run algorithms on test data
    model.transform(testData)
      .show()

  }

}

/*

Assignment: Stack Exchange post classification:
1. Using labeled seed data (seed.csv), do model training and evaluation to predict a post category based
on its body. Use the following algorithms:
  1. Logistic Regression
  2. Decision Tree
  3. Naive Bayes
  4. Artificial Neural Network (Perceptron)
2. Divide seed data into 70% training / 30% validation set, while training.
3. Then, use input_data.csv file (not labeled) as your test set and predict their category
4. Please upload your solution to your personal github with instructions on how to run it.

Notice that prediction quality would be very low due to the small seed data, we are more interested in your data problem
solving skills.



 */
