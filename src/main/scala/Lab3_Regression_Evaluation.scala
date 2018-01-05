import org.apache.spark.sql.SparkSession
// Import Spark SQL and Spark ML libraries
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator

object Lab3_Regression_Evaluation {
  def main(args: Array[String]) {
    //This section is needed to be able to run against your local cluster
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Lab 3 Regression Evaluation")
      .getOrCreate()

//    Evaluating a Regression Model¶
//    In this exercise, you will create a pipeline for a linear regression model, and then test and evaluate the model.
//
//      Prepare the Data
//    First, import the libraries you will need and prepare the training and test data:

    // Load the source data
    val csv = spark.read.option("inferSchema","true").option("header", "true").csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/flights.csv")

    // Select features and label
    val data = csv.select(col("DayofMonth"), col("DayOfWeek"), col("OriginAirportID"), col("DestAirportID"), col("DepDelay"), col("ArrDelay").alias("label"))

    // Split the data
    val splits = data.randomSplit(Array(0.7, 0.3))
    val train = splits(0)
    val test = splits(1).withColumnRenamed("label", "trueLabel")

//    Define the Pipeline and Train the Model
//    Now define a pipeline that creates a feature vector and trains a regression model

    // Define the pipeline
    val assembler = new VectorAssembler().setInputCols(Array("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay")).setOutputCol("features")
    val lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)
    val pipeline = new Pipeline().setStages(Array(assembler, lr))

    // Train the model
    val model = pipeline.fit(train)

//    Test the Model¶
//    Now you're ready to apply the model to the test data.

    val prediction = model.transform(test)
    val predicted = prediction.select("features", "prediction", "trueLabel")
    predicted.show()

//    Examine the Predicted and Actual Values¶
//    You can plot the predicted values against the actual values to see how accurately the model has predicted. In a perfect model, the resulting scatter plot should form a perfect diagonal line with each predicted value being identical to the actual value - in practice, some variance is to be expected. Run the cells below to create a temporary table from the predicted DataFrame and then retrieve the predicted and actual label values using SQL. You can then display the results as a scatter plot, specifying - as the function to show the unaggregated values.

    predicted.createOrReplaceTempView("regressionPredictions")

//    Retrieve the Root Mean Square Error (RMSE)¶
//    There are a number of metrics used to measure the variance between predicted and actual values. Of these, the root mean square error (RMSE) is a commonly used value that is measured in the same units as the predicted and actual values - so in this case, the RMSE indicates the average number of minutes between predicted and actual flight delay values. You can use the RegressionEvaluator class to retrieve the RMSE.

    val evaluator = new RegressionEvaluator().setLabelCol("trueLabel").setPredictionCol("prediction").setMetricName("rmse")
    val rmse = evaluator.evaluate(prediction)
    println("Root Mean Square Error (RMSE): " + (rmse))

  }
}
