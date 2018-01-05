import org.apache.spark.sql.SparkSession

//Tuning Model Parameters
//In this exercise, you will optimise the parameters for a classification model.
//
//  Prepare the Data
//First, import the libraries you will need and prepare the training and test data:

// Import Spark SQL and Spark ML libraries
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.functions._

object Lab3_Tuning_Parameters {
  def main(args: Array[String]) {
    //This section is needed to be able to run against your local cluster
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Lab 3 Regression Evaluation")
      .getOrCreate()

    // Load the source data
    val csv = spark.read.option("inferSchema","true").option("header", "true").csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/flights.csv")

    // Select features and label
    val data = csv.select(col("DayofMonth"), col("DayOfWeek"), col("OriginAirportID"), col("DestAirportID"), col("DepDelay"), (col("ArrDelay") > 15).cast("Int").alias("label"))

    // Split the data
    val splits = data.randomSplit(Array(0.7, 0.3))
    val train = splits(0)
    val test = splits(1).withColumnRenamed("label", "trueLabel")

//    Define the Pipeline
//    Now define a pipeline that creates a feature vector and trains a classification model

    val assembler = new VectorAssembler().setInputCols(Array("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay")).setOutputCol("features")
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
    val pipeline = new Pipeline().setStages(Array(assembler, lr))

//    Tune Parameters
//    You can tune parameters to find the best model for your data. A simple way to do this is to use TrainValidationSplit to evaluate each combination of parameters defined in a ParameterGrid against a subset of the training data in order to find the best performing parameters

    val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.3, 0.1, 0.01)).addGrid(lr.maxIter, Array(10, 5)).addGrid(lr.threshold, Array(0.35, 0.3)).build()
    val tvs = new TrainValidationSplit().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(paramGrid).setTrainRatio(0.8)

    val model = tvs.fit(train)
//
//    Test the Model¶
//    Now you're ready to apply the model to the test data.

    val prediction = model.transform(test)
    val predicted = prediction.select("features", "prediction", "probability", "trueLabel")
    predicted.show(100)
//
//    Compute Confusion Matrix Metrics
//    Now you can examine the confusion matrix metrics to judge the performance of the model.

    val tp = predicted.filter("prediction == 1 AND truelabel == 1").count().toFloat
    val fp = predicted.filter("prediction == 1 AND truelabel == 0").count().toFloat
    val tn = predicted.filter("prediction == 0 AND truelabel == 0").count().toFloat
    val fn = predicted.filter("prediction == 0 AND truelabel == 1").count().toFloat
    val metrics = spark.createDataFrame(Seq(
      ("TP", tp),
      ("FP", fp),
      ("TN", tn),
      ("FN", fn),
      ("Precision", tp / (tp + fp)),
      ("Recall", tp / (tp + fn)))).toDF("metric", "value")
    metrics.show()

//    Review the Area Under ROC¶
//    You can also assess the accuracy of the model by reviewing the area under ROC metric.

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("trueLabel").setRawPredictionCol("prediction").setMetricName("areaUnderROC")
    val aur = evaluator.evaluate(prediction)
    println("AUR = " + (aur))
  }
}
