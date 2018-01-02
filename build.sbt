name := "SparkDemo"

version := "0.1"

scalaVersion := "2.11.4"

libraryDependencies ++= {
  val sparkVer = "2.2.1"
  Seq(
    "org.apache.spark" %% "spark-core" % sparkVer,
    "org.apache.spark" %% "spark-sql" % sparkVer
  )
}

resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"