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


  }
}
