import org.apache.spark.sql.SparkSession


//Evaluating a Classification Model
//  In this exercise, you will create a pipeline for a classification model, and then apply commonly used metrics to evaluate the resulting classifier.
//
//  Prepare the Data
//First, import the libraries you will need and prepare the training and test data:

// Import Spark SQL and Spark ML libraries
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object Lab3_Classification_Evaluation {
  def main(args: Array[String]) {
    //This section is needed to be able to run against your local cluster
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Lab 3 Classification Evaluation")
      .getOrCreate()


    // Load the source data
    val csv = spark.read.option("inferSchema","true").option("header", "true").csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/flights.csv")

    // Select features and label
    val data = csv.select(col("DayofMonth"), col("DayOfWeek"), col("OriginAirportID"), col("DestAirportID"), col("DepDelay"), (col("ArrDelay") > 15).cast("Int").alias("label"))

    // Split the data
    val splits = data.randomSplit(Array(0.7, 0.3))
    val train = splits(0)
    val test = splits(1).withColumnRenamed("label", "trueLabel")

//    Define the Pipeline and Train the Model¶
//    Now define a pipeline that creates a feature vector and trains a classification model

    // Define the pipeline
    val assembler = new VectorAssembler().setInputCols(Array("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay")).setOutputCol("features")
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)
    val pipeline = new Pipeline().setStages(Array(assembler, lr))

    // Train the model
    val model = pipeline.fit(train)

//    Test the Model
//    Now you're ready to apply the model to the test data.


    val prediction = model.transform(test)
    val predicted = prediction.select("features", "prediction", "trueLabel")
    predicted.show(100, truncate=false)

//    Compute Confusion Matrix Metrics
//    Classifiers are typically evaluated by creating a confusion matrix, which indicates the number of:
//
//    True Positives
//    True Negatives
//    False Positives
//    False Negatives
//    From these core measures, other evaluation metrics such as precision and recall can be calculated.

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

//    View the Raw Prediction and Probability
//    The prediction is based on a raw prediction score that describes a labelled point in a logistic function. This raw prediction is then converted to a predicted label of 0 or 1 based on a probability vector that indicates the confidence for each possible label value (in this case, 0 and 1). The value with the highest confidence is selected as the prediction.

    prediction.select("rawPrediction", "probability", "prediction", "trueLabel").show(100, truncate=false)

//    Note that the results include rows where the probability for 0 (the first value in the probability vector) is only slightly higher than the probability for 1 (the second value in the probability vector). The default discrimination threshold (the boundary that decides whether a probability is predicted as a 1 or a 0) is set to 0.5; so the prediction with the highest probability is always used, no matter how close to the threshold.
//
//    Review the Area Under ROC
//    Another way to assess the performance of a classification model is to measure the area under a ROC curve for the model. the spark.ml library includes a BinaryClassificationEvaluator class that you can use to compute this. The ROC curve shows the True Positive and False Positive rates plotted for varying thresholds.

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("trueLabel").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
    val auc = evaluator.evaluate(prediction)
    println("AUC = " + (auc))

//    Change the Discrimination Threshold
//      The AUC score seems to indicate a reasonably good model, but the performance metrics seem to indicate that it predicts a high number of False Negative labels (i.e. it predicts 0 when the true label is 1), leading to a low Recall. You can affect the way a model performs by changing its parameters. For example, as noted previously, the default discrimination threshold is set to 0.5 - so if there are a lot of False Positives, you may want to consider raising this; or conversely, you may want to address a large number of False Negatives by lowering the threshold.

    // Redefine the pipeline
    val lr2 = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setThreshold(0.35).setMaxIter(10).setRegParam(0.3)
    val pipeline2 = new Pipeline().setStages(Array(assembler, lr2))

    // Retrain the model
    val model2 = pipeline2.fit(train)
    // Retest the model
    val newPrediction = model2.transform(test)
    newPrediction.select("rawPrediction", "probability", "prediction", "trueLabel").show(100, truncate=false)

//    Note that some of the rawPrediction and probability values that were previously predicted as 0 are now predicted as 1

    // Recalculate confusion matrix
    val tp2 = newPrediction.filter("prediction == 1 AND truelabel == 1").count().toFloat
    val fp2 = newPrediction.filter("prediction == 1 AND truelabel == 0").count().toFloat
    val tn2 = newPrediction.filter("prediction == 0 AND truelabel == 0").count().toFloat
    val fn2 = newPrediction.filter("prediction == 0 AND truelabel == 1").count().toFloat
    val metrics2 = spark.createDataFrame(Seq(
      ("TP", tp2),
      ("FP", fp2),
      ("TN", tn2),
      ("FN", fn2),
      ("Precision", tp2 / (tp2 + fp2)),
      ("Recall", tp2 / (tp2 + fn2)))).toDF("metric", "value")
    metrics2.show()

    //    Note that there are now more True Positives and less False Negatives, and Recall has improved. By changing the discrimination threshold, the model now gets more predictions correct - though it's worth noting that the number of False Positives has also increased.

  }
}
