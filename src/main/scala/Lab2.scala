import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler

object Lab2 {

  def main(args: Array[String]) {
    //This section is needed to be able to run against your local cluster
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Lab 2 Supervised learning models")
      .getOrCreate()

//    Creating a Classification Model¶
//    In this exercise, you will implement a classification model that uses features of a flight to predict whether or not the flight will be delayed.
//
//    Import Spark SQL and Spark ML Libraries
//    First, import the libraries you will need: see imports above

//    Load Source Data¶
//    The data for this exercise is provided as a CSV file containing details of flights. The data includes specific characteristics (or features) for each flight, as well as a column indicating how many minutes late or early the flight arrived.
//
//      You will load this data into a DataFrame and display it.
    val csv = spark.read.option("inferSchema","true").option("header", "true").csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/flights.csv")
    csv.show()

//    Prepare the Data
//    Most modeling begins with exhaustive exploration and preparation of the data. In this example, the data has been cleaned for you. You will simply select a subset of columns to use as features and create a Boolean label field named Late with the value 1 for flights that arrived 15 minutes or more after the scheduled arrival time, or 0 if the flight was early or on-time.
//
//    (Note that in a real scenario, you would perform additional tasks such as handling missing or duplicated data, scaling numeric columns, and using a process called feature engineering to create new features for your model).

    val data = csv.select(csv.col("DayofMonth"), csv.col("DayOfWeek"), csv.col("OriginAirportID"), csv.col("DestAirportID"), csv.col("DepDelay"), (csv.col("ArrDelay") > 15).cast("Int").alias("Late"))
    data.show()

//    Split the Data¶
//    It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this exercise, you will use 70% of the data for training, and reserve 30% for testing.

    val splits = data.randomSplit(Array(0.7, 0.3))
    val train = splits(0)
    val test = splits(1)
    val train_rows = train.count()
    val test_rows = test.count()
    println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

//    Prepare the Training Data
//      To train the classification model, you need a training data set that includes a vector of numeric features, and a label column. In this exercise, you will use the VectorAssembler class to transform the feature columns into a vector, and then rename the Late column to label.
    val assembler = new VectorAssembler().setInputCols(Array("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay")).setOutputCol("features")
    val training = assembler.transform(train).select(col("features"), col("Late").alias("label"))

    training.show()
  }
}