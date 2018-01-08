import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

object Lab4_Collaborative_Filtering {
  def main(args: Array[String]) {
    //This section is needed to be able to run against your local cluster
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Lab 4 Collaborative filtering")
      .getOrCreate()
//
//    Load Source Data¶
//    The source data for the recommender is in two files - one containing numeric IDs for movies and users, along with user ratings; and the other containing details of the movies.

    val movies = spark.read.option("inferSchema","true").option("header", "true").csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/movies.csv")
    val ratings = spark.read.option("inferSchema","true").option("header", "true").csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/ratings.csv")
    ratings.join(movies, "movieId").show()

//    Prepare the Data¶
//    To prepare the data, split it into a training set and a test set.

    val data = ratings.select("userId", "movieId", "rating")
    val splits = data.randomSplit(Array(0.7, 0.3))
    val train = splits(0).withColumnRenamed("rating", "label")
    val test = splits(1).withColumnRenamed("rating", "trueLabel")
    val train_rows = train.count()
    val test_rows = test.count()
    println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

//    Build the Recommender
//    The ALS class is an estimator, so you can use its fit method to traing a model, or you can include it in a pipeline. Rather than specifying a feature vector and as label, the ALS algorithm requries a numeric user ID, item ID, and rating.


    val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("movieId").setRatingCol("label")
    val model = als.fit(train)

//    Test the Recommender
//    Now that you've trained the recommender, you can see how accurately it predicts known ratings in the test set.

    val prediction = model.transform(test)
    prediction.join(movies, "movieId").select("userId", "title", "prediction", "trueLabel").show(100, truncate=false)

//    The data used in this exercise describes 5-star rating activity from MovieLens, a movie recommendation service. It was created by GroupLens, a research group in the Department of Computer Science and Engineering at the University of Minnesota, and is used here with permission.
//
//      This dataset and other GroupLens data sets are publicly available for download at http://grouplens.org/datasets/.
//
//    For more information, see F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015)
  }
}
