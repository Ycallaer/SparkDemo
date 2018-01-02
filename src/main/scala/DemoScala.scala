import org.apache.spark.sql.SparkSession

object DemoScala {
  def main(args: Array[String]) {

    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Spark SQL basic example")
      .getOrCreate()

    val df=spark.read.csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/airports.csv")
    df.show()
  }
}
