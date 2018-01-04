import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler

object Lab2_Regression {
  def main(args: Array[String]) {
    //This section is needed to be able to run against your local cluster
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Lab 2 Supervised learning models: Regression")
      .getOrCreate()

//    Creating a Regression Model
//      In this exercise, you will implement a regression model that uses features of a flight to predict how late or early it will arrive.
//
//      Import Spark SQL and Spark ML Libraries
//    First, import the libraries you will need: see above


//    Load Source Data¶
//    The data for this exercise is provided as a CSV file containing details of flights. The data includes specific characteristics (or features) for each flight, as well as a column indicating how many minutes late or early the flight arrived.
//
//      You will load this data into a DataFrame and display it.

    val csv = spark.read.option("inferSchema","true").option("header", "true").csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/flights.csv")
    csv.show()

//    Prepare the Data
//    Most modeling begins with exhaustive exploration and preparation of the data. In this example, you will simply select a subset of columns to use as features as well as the ArrDelay column, which will be the label your model will predict.

    val data = csv.select(col("DayofMonth"), col("DayOfWeek"), col("OriginAirportID"), col("DestAirportID"), col("DepDelay"), col("ArrDelay"))
    data.show()

//    Split the Data
//    It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this exercise, you will use 70% of the data for training, and reserve 30% for testing.

    val splits = data.randomSplit(Array(0.7, 0.3))
    val train = splits(0)
    val test = splits(1)
    val train_rows = train.count()
    val test_rows = test.count()
    println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

//    Prepare the Training Data
//      To train the regression model, you need a training data set that includes a vector of numeric features, and a label column. In this exercise, you will use the VectorAssembler class to transform the feature columns into a vector, and then rename the ArrDelay column to label.
    val assembler = new VectorAssembler().setInputCols(Array("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay")).setOutputCol("features")
    val training = assembler.transform(train).select(col("features"), col("ArrDelay").cast("Int").alias("label"))
    training.show()

//    Train a Regression Model¶
//    Next, you need to train a regression model using the training data. To do this, create an instance of the regression algorithm you want to use and use its fit method to train a model based on the training DataFrame. In this exercise, you will use a Linear Regression algorithm - though you can use the same technique for any of the regression algorithms supported in the spark.ml API.
    val lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)
    val model = lr.fit(training)
    println("Model Trained!")

//    Prepare the Testing Data
//      Now that you have a trained model, you can test it using the testing data you reserved previously. First, you need to prepare the testing data in the same way as you did the training data by transforming the feature columns into a vector. This time you'll rename the ArrDelay column to trueLabel.

    val testing = assembler.transform(test).select(col("features"), col("ArrDelay").cast("Int").alias("trueLabel"))
    testing.show()

//    Test the Model¶
//    Now you're ready to use the transform method of the model to generate some predictions. You can use this approach to predict arrival delay for flights where the label is unknown; but in this case you are using the test data which includes a known true label value, so you can compare the predicted number of minutes late or early to the actual arrival delay.

    val prediction = model.transform(testing)
    val predicted = prediction.select("features", "prediction", "trueLabel")
    predicted.show()
  }
}
