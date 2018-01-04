import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.SparkSession

object Lab2_Text_Analysis {
  def main(args: Array[String]) {
    //This section is needed to be able to run against your local cluster
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Lab 2 Supervised learning models: Regression")
      .getOrCreate()

//    Load Source Data
//    Now load the tweets data into a DataFrame. This data consists of tweets that have been previously captured and classified as positive or negative.

    val tweets_csv = spark.read.option("inferSchema","true").option("header", "true").csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/tweets.csv")
    tweets_csv.show(truncate = false)

//    Prepare the Data
//    The features for the classification model will be derived from the tweet text. The label is the sentiment (1 for positive, 0 for negative)

    val data = tweets_csv.select(col("SentimentText"), col("Sentiment").cast("Int").alias("label"))
    data.show(truncate = false)
//    Split the Data
//    In common with most classification modeling processes, you'll split the data into a set for training, and a set for testing the trained model.

    val splits = data.randomSplit(Array(0.7, 0.3))
    val train = splits(0)
    val test = splits(1).withColumnRenamed("label", "trueLabel")
    val train_rows = train.count()
    val test_rows = test.count()
    println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

//    Define the Pipeline
//    The pipeline for the model consist of the following stages:
//
//      A Tokenizer to split the tweets into individual words.
//      A StopWordsRemover to remove common words such as "a" or "the" that have little predictive value.
//    A HashingTF class to generate numeric vectors from the text values.
//      A LogisticRegression algorithm to train a binary classification model.
//

    val tokenizer = new Tokenizer().setInputCol("SentimentText").setOutputCol("SentimentWords")
    val swr = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("MeaningfulWords")
    val hashTF = new HashingTF().setInputCol(swr.getOutputCol).setOutputCol("features")
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.01)
    val pipeline = new Pipeline().setStages(Array(tokenizer, swr, hashTF, lr))

//    Run the Pipeline as an Estimator
//      The pipeline itself is an estimator, and so it has a fit method that you can call to run the pipeline on a specified DataFrame. In this case, you will run the pipeline on the training data to train a model.


    val piplineModel = pipeline.fit(train)
    println("Pipeline complete!")

//    Test the Pipeline Model
//      The model produced by the pipeline is a transformer that will apply all of the stages in the pipeline to a specified DataFrame and apply the trained model to generate predictions. In this case, you will transform the test DataFrame using the pipeline to generate label predictions.


    val prediction = piplineModel.transform(test)
    val predicted = prediction.select("SentimentText", "features", "prediction", "trueLabel")
    predicted.show(100, truncate = false)

  }
}
