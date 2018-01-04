import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler}
import org.apache.spark.ml.classification.DecisionTreeClassifier

object Lab2_Pipeline {
  def main(args: Array[String]) {
    //This section is needed to be able to run against your local cluster
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Lab 2 Supervised learning models: Regression")
      .getOrCreate()

//    Creating a Pipeline
//    In this exercise, you will implement a pipeline that includes multiple stages of transformers and estimators to prepare features and train a classification model. The resulting trained PipelineModel can then be used as a transformer to predict whether or not a flight will be late.
//
//      Import Spark SQL and Spark ML Libraries
//    First, import the libraries you will need: see above

//
//    Load Source Data¶
//    The data for this exercise is provided as a CSV file containing details of flights. The data includes specific characteristics (or features) for each flight, as well as a column indicating how many minutes late or early the flight arrived.
//
//      You will load this data into a DataFrame and display it.

    val csv = spark.read.option("inferSchema","true").option("header", "true").csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/flights.csv")
    csv.show()

//    Prepare the Data
//    Most modeling begins with exhaustive exploration and preparation of the data. In this example, the data has been cleaned for you. You will simply select a subset of columns to use as features and create a Boolean label field named label with the value 1 for flights that arrived 15 minutes or more after the scheduled arrival time, or 0 if the flight was early or on-time.

    val data = csv.select(col("DayofMonth"), col("DayOfWeek"), col("Carrier"), col("OriginAirportID"), col("DestAirportID"), col("DepDelay"), (col("ArrDelay") > 15).cast("Double").alias("label"))
    data.show()

//    Split the Data
//    It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this exercise, you will use 70% of the data for training, and reserve 30% for testing. In the testing data, the label column is renamed to trueLabel so you can use it later to compare predicted labels with known actual values.
    val splits = data.randomSplit(Array(0.7, 0.3))
    val train = splits(0)
    val test = splits(1).withColumnRenamed("label", "trueLabel")
    val train_rows = train.count()
    val test_rows = test.count()
    println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)


//    Define the Pipeline¶
//    A predictive model often requires multiple stages of feature preparation. For example, it is common when using some algorithms to distingish between continuous features (which have a calculable numeric value) and categorical features (which are numeric representations of discrete categories). It is also common to normalize continuous numeric features to use a common scale (for example, by scaling all numbers to a proportinal decimal value between 0 and 1).
//    A pipeline consists of a a series of transformer and estimator stages that typically prepare a DataFrame for modeling and then train a predictive model. In this case, you will create a pipeline with seven stages:
//    A StringIndexer estimator that converts string values to indexes for categorical features
//    A VectorAssembler that combines categorical features into a single vector
//    A VectorIndexer that creates indexes for a vector of categorical features
//    A VectorAssembler that creates a vector of continuous numeric features
//    A MinMaxScaler that normalizes continuous numeric features
//    A VectorAssembler that creates a vector of categorical and continuous features
//    A DecisionTreeClassifier that trains a classification model.

    val strIdx = new StringIndexer().setInputCol("Carrier").setOutputCol("CarrierIdx")
    val catVect = new VectorAssembler().setInputCols(Array("CarrierIdx", "DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID")).setOutputCol("catFeatures")
    val catIdx = new VectorIndexer().setInputCol(catVect.getOutputCol).setOutputCol("idxCatFeatures")
    val numVect = new VectorAssembler().setInputCols(Array("DepDelay")).setOutputCol("numFeatures")
    val minMax = new MinMaxScaler().setInputCol(numVect.getOutputCol).setOutputCol("normFeatures")
    val featVect = new VectorAssembler().setInputCols(Array("idxCatFeatures", "normFeatures")).setOutputCol("features")
    val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")
    val pipeline = new Pipeline().setStages(Array(strIdx, catVect, catIdx, numVect, minMax, featVect, dt))

//    Run the Pipeline as an Estimator
//      The pipeline itself is an estimator, and so it has a fit method that you can call to run the pipeline on a specified DataFrame. In this case, you will run the pipeline on the training data to train a model.

    val model = pipeline.fit(train)
    println("Pipeline complete!")
//
//    Test the Pipeline Model
//      The model produced by the pipeline is a transformer that will apply all of the stages in the pipeline to a specified DataFrame and apply the trained model to generate predictions. In this case, you will transform the test DataFrame using the pipeline to generate label predictions.

    val prediction = model.transform(test)
    val predicted = prediction.select("features", "prediction", "trueLabel")
    predicted.show(100, truncate=false)

//    The resulting DataFrame is produced by applying all of the transformations in the pipline to the test data. The prediction column contains the predicted value for the label, and the trueLabel column contains the actual known value from the testing data.



  }
}
