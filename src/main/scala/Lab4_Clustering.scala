import org.apache.spark.sql.SparkSession

//Clustering¶
//In this exercise, you will use K-Means clustering to segment customer data into five clusters.
//
//  Import the Libraries
//You will use the KMeans class to create your model. This will require a vector of features, so you will also use the VectorAssembler class.

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler

object Lab4_Clustering {
  def main(args: Array[String]) {
    //This section is needed to be able to run against your local cluster
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Lab 4 Clustering")
      .getOrCreate()
//
//    Load Source Data¶
//    The source data for your clusters is in a comma-separated values (CSV) file, and incldues the following features:
//
//    CustomerName: The custome's name
//    Age: The customer's age in years
//    MaritalStatus: The custtomer's marital status (1=Married, 0 = Unmarried)
//    IncomeRange: The top-level for the customer's income range (for example, a value of 25,000 means the customer earns up to 25,000)
//    Gender: A numeric value indicating gender (1 = female, 2 = male)
//    TotalChildren: The total number of children the customer has
//    ChildrenAtHome: The number of children the customer has living at home.
//    Education: A numeric value indicating the highest level of education the customer has attained (1=Started High School to 5=Post-Graduate Degree
//    Occupation: A numeric value indicating the type of occupation of the customer (0=Unskilled manual work to 5=Professional)
//    HomeOwner: A numeric code to indicate home-ownership (1 - home owner, 0 = not a home owner)
//    Cars: The number of cars owned by the customer.

    val customers = spark.read.option("inferSchema","true").option("header", "true").csv("/Users/yvescallaert/Documents/DAT202.3x/Lab_Files/DAT202.3x/data/customers.csv")
    customers.show()

//    Create the K-Means Model
//    You will use the feaures in the customer data to create a Kn-Means model with a k value of 5. This will be used to generate 5 clusters

    val assembler = new VectorAssembler().setInputCols(Array("Age", "MaritalStatus", "IncomeRange", "Gender", "TotalChildren", "ChildrenAtHome", "Education", "Occupation", "HomeOwner", "Cars")).setOutputCol("features")
    val train = assembler.transform(customers)

    val kmeans = new KMeans().setFeaturesCol(assembler.getOutputCol).setPredictionCol("cluster").setK(5).setSeed(0)
    val model = kmeans.fit(train)
    println("Model Created!")
//
//    Get the Cluster Centers
//    The cluster centers are indicated as vector coordinates.

    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)

//    Predict Clusters¶
//    Now that you have trained the model, you can use it to segemnt the customer data into 5 clusters and show each customer with their allocated cluster.

    val prediction = model.transform(train)
    prediction.groupBy("cluster").count().orderBy("cluster").show()
    prediction.select("CustomerName", "cluster").show(50)
  }
}
