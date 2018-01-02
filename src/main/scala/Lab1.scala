import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Encoders

//Case class had to be moved outside, otherwise you get an error
case class flight(DayofMonth:Int, DayOfWeek:Int, Carrier:String, OriginAirportID:Int, DestAirportID:Int, DepDelay:Int, ArrDelay:Int)


object DemoScala {
  def main(args: Array[String]) {
    //This section is needed to be able to run against your local cluster
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Spark SQL basic example")
      .getOrCreate()

    //Load Data Using an Explicit Schema
    //  To explore data, you must load it into a programmatic data object such as a DataFrame. If the structure of the data is known ahead of time, you can explicitly specify the schema for the DataFrame.

    //In this exercise, you will work with data that records details of flights.
    val flightSchema = Encoders.product[flight].schema

    val flights = spark.read.schema(flightSchema).option("header", "true").csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/raw-flight-data.csv")
    flights.show()

    //Infer a Data Schema
    //If the structure of the data source is unknown, you can have Spark auomatically infer the schema.

    //In this case, you will load data about airports without knowing the schema.
    val airports = spark.read.option("inferSchema","true").option("header","true").csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/airports.csv")
    airports.show()

    //Use DataFrame Methods
    //Spark DataFrames provide functions that you can use to extract and manipulate data. For example, you can use the select function to return a new DataFrame containing columns selected from an existing DataFrame.
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

    //Check Summary Statistics
    //After cleaning the data, you should re-check the statistics - removing rows and changing values may affect the distribution of the data, which in turn could affect any predictive models you might create
    data.describe().show()

    //Explore Relationships in the Data
    //Predictive modeling is largely based on statistical relationships between fields in the data. To design a good model, you need to understand how the data points relate to one another and identify any apparent correlation. The stat.corr function calculates a correlation value between -1 and 1, indicating the strength of correlation between two fields. A strong positive correlation (near 1) indicates that high values for one column are often found with high values for the other, which a string negative correlation (near -1) indicates that low values for one column are often found with high values for the other. A correlation near 0 indicates little apparent relationship between the fields.

    data.stat.corr("DepDelay", "ArrDelay")

    //Use Spark SQL¶
    //In addition to using the DataFrame API directly to query data, you can persist DataFrames as table and use Spark SQL to query them using the SQL language. SQL is often more intuitive to use when querying tabular data structures.
    data.createOrReplaceTempView("flightData")
    spark.sql("SELECT DayOfWeek, AVG(ArrDelay) AS AvgDelay FROM flightData GROUP BY DayOfWeek ORDER BY DayOfWeek").show()
  }
}
