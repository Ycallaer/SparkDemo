import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Encoders

case class flight(DayofMonth:Int, DayOfWeek:Int, Carrier:String, OriginAirportID:Int, DestAirportID:Int, DepDelay:Int, ArrDelay:Int)


object DemoScala {
  def main(args: Array[String]) {

    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Spark SQL basic example")
      .getOrCreate()

    val flightSchema = Encoders.product[flight].schema

    val flights = spark.read.schema(flightSchema).option("header", "true").csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/raw-flight-data.csv")
    flights.show()

    val airports = spark.read.option("inferSchema","true").option("header","true").csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/airports.csv")
    airports.show()

    val cities = airports.select("city", "name")
    cities.show()

    val flightsByOrigin= flights.join(airports,flights.col("OriginAirportID") === airports.col("airport_id")).groupBy("city").count()
    flightsByOrigin.show()

    //Determine Summary Statistics¶
    //Predictive modeling is based on statistics and probability, so you will often start by looking at summary statistics. The describe function returns a DataFrame containing the count, mean, standard deviation, minimum, and maximum values for each numeric column.

    flights.describe().show()

    //Determine the Presence of Duplicates
    //The data you have to work with won't always be perfect - often you'll want to clean the data; for example to detect and remove duplicates that might affect your model. You can use the dropDuplicates function to create a new DataFrame with the duplicates removed, enabling you to determine how many rows are duplicates of other rows.

    flights.count() - flights.dropDuplicates().count()

    //Identify Missing Values¶
    //As well as determing if duplicates exist in your data, you should detect missing values, and either remove rows containing missing data or replace the missing values with a suitable relacement. The na.drop function creates a DataFrame with any rows containing missing data removed - you can specify a subset of columns, and whether the row should be removed in any or all values are missing. You can then use this new DataFrame to determine how many rows contain missing values.
    flights.count() - flights.dropDuplicates().na.drop("any", Array("ArrDelay", "DepDelay")).count()

    //Clean the Data¶
    //Now that you've identified that there are duplicates and missing values, you can clean the data by removing the duplicates and replacing the missing values. The na.fill function replaces missing values with a specified replacement value. In this case, you'll remove all duplicate rows and replace missing ArrDelay and DepDelay values with 0.

    val data=flights.dropDuplicates().na.fill(0, Array("ArrDelay", "DepDelay"))
    data.count()

  }
}
