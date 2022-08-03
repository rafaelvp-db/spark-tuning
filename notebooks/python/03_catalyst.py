# Databricks notebook source
# MAGIC %md # Catalyst
# MAGIC 
# MAGIC ## Catalyst Anti-patterns
# MAGIC 
# MAGIC We will begin by reviewing 3 common anti-patterns that can hurt application performance and prevent Catalyst optimizations:
# MAGIC - Partially cached DFs 
# MAGIC - User defined functions
# MAGIC - Cartesian products
# MAGIC 
# MAGIC ### Partially cached DataFrames 
# MAGIC 
# MAGIC A partially cached DataFrame is considered an anti-pattern, as re-computation of missing partitions can be expensive. If any of the transformations carried out on missing data were wide (they required a shuffle where data was moved from one stage to another), all the work on the missing partition would have to be redone.
# MAGIC 
# MAGIC To attempt to avoid this scenario, default settings for caching a `DataFrame` or `Dataset` have been altered compared to the older settings used for `RDD`s. By default the `StorageLevel.MEMORY_AND_DISK_SER` level is used for a DF's `cache()` function.

# COMMAND ----------

from pyspark import StorageLevel
from pyspark.sql.types import *
people_schema = StructType([
  StructField("id", IntegerType(), True),
  StructField("firstName", StringType(), True),
  StructField("middleName", StringType(), True),
  StructField("lastName", StringType(), True),
  StructField("gender", StringType(), True),
  StructField("birthDate", TimestampType(), True),
  StructField("ssn", StringType(), True),
  StructField("salary", IntegerType(), True)
])

# code used to generate screenshot #1 below (partial cache).
people = (
  spark
  .read
  .option("header", "true")
  .option("delimiter", ":")
  .schema(people_schema)
  .csv("/mnt/training/dataframes/people-with-header-10m.txt")
)

people2 = people.union(people)
people3 = people2.union(people2)
people4 = people3.union(people3)
people4.persist(StorageLevel.MEMORY_ONLY).count()

# COMMAND ----------

people4.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Partial DF Caching
# MAGIC 
# MAGIC Any time further processing needs to be carried out on a partially cached DataFrame as above, it will cause re-computation that would involve pulling data from the initial data source. A simple solution is to use a `StorageLevel` that will save any data that doesn't fit in memory on to disk instead, such as `StorageLevel.MEMORY_AND_DISK_SER`. Note that the default implementation of `cache()`, for DataFrames, properly ensures that the entire DataFrame will be cached, even if the data has to spill to local disk.
# MAGIC 
# MAGIC ## Full DF Caching
# MAGIC 
# MAGIC Accessing data from disk is slower than memory, but, in this case, the disk is _local_ to the node, so it's likely to be faster than reading from the original data source. Plus, it's better to avoid re-computation, especially if any shuffling (due to wide transformations) is involved.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### User defined functions
# MAGIC 
# MAGIC UDFs require deserialization of data stored under the Tungsten format. The data needs to be available as an object in an executor so the UDF function can be applied to it. 
# MAGIC 
# MAGIC Below is an example of a UDF that would require a column value to be deserialized from Tungsten's format to an object, to allow the UDF to operate on it.

# COMMAND ----------

from pyspark.sql.functions import udf

upperUDF = udf(lambda s: s.upper())
lowerUDF = udf(lambda s: s.lower())

initDF = spark.read.parquet("dbfs:/mnt/training/dataframes/people-10m.parquet")
udfDF = initDF.select(upperUDF(initDF["firstName"]), lowerUDF(initDF["middleName"]), upperUDF(initDF["lastName"]), lowerUDF(initDF["gender"]))
udfDF.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC But:
# MAGIC 
# MAGIC - Numerous utility functions for DataFrames and Datasets are already available in spark. 
# MAGIC - These functions are located in the <a href="https://spark.apache.org/docs/2.0.0/api/java/org/apache/spark/sql/functions.html">functions package</a> in spark under `org.apache.spark.sql.functions`
# MAGIC 
# MAGIC Using built in function usage is preferred over coding UDFs:
# MAGIC - As built-in functions are integrated with Catalyst, they can be optimized in ways in which UDFs cannot. 
# MAGIC - Built-in functions can benefit from code-gen and can also manipulate our dataset even when it's serialized using the Tungsten format without `serialization / deserialization` overhead.
# MAGIC - Python UDFs carry extra overhead as they require additional serialization from the driver vm to the executor's JVM. 
# MAGIC     - A Hive UDF can be used instead of a python UDF. This will avoid serialization overheads. 
# MAGIC  
# MAGIC Below is an example of using the built in `lower` and `upper` functions. 

# COMMAND ----------

from pyspark.sql.functions import lower, upper

noUDF = initDF.select(upper(initDF["firstName"]), lower(initDF["middleName"]), upper(initDF["lastName"]), lower(initDF["gender"]))
noUDF.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Cartesian Products
# MAGIC 
# MAGIC Put simply, a Cartesian product is a set that contains all possible combinations of elements from two other sets. Related to Spark SQL this can be a table that contains all combinations of rows from two other tables, that were joined together.
# MAGIC 
# MAGIC Cartesian products are problematic as they are a sign of an expensive computation.
# MAGIC 
# MAGIC First, let's force the broadcast join threshold very low, just to ensure no side of the join is broadcast. (We do this for demo purposes.)

# COMMAND ----------

previous_threshold = spark.conf.get("spark.sql.autoBroadcastJoinThreshold")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "0")

# COMMAND ----------

numbDF = spark.createDataFrame(map(lambda x: (x, ), range(1, 4)), ["n1"])
numbDF2 = spark.createDataFrame(map(lambda x: (x, ), range(4, 7)), ["n2"])

cartesianDF = numbDF.crossJoin(numbDF2)
cartesianDF.explain()

# COMMAND ----------

# MAGIC %md Let's reset the `spark.sql.autoBroadcastJoinThreshold` value to its default.

# COMMAND ----------

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", previous_threshold)
print("Restored broadcast join threshold to {0}".format(spark.conf.get("spark.sql.autoBroadcastJoinThreshold")))

# COMMAND ----------

# MAGIC %md `BroadcastNestedLoopJoin` is the result of a Cartesian product that contains a DataFrame small enough to be broadcasted.
# MAGIC `CartesianProduct` would be the result of a Cartesian join where neither DataFrame is small enough to be broadcasted.
# MAGIC 
# MAGIC **Don't run an action that requires accessing all partitions on the below Cartesian product**. It can take a very long time!
# MAGIC 
# MAGIC `first()` is fine, because it's only going to operate on the first partition.

# COMMAND ----------

ipDF = spark.read.parquet("/mnt/training/ip-geocode.parquet")
print("ipDF.count: " + str(ipDF.count()))

cartesianDF2 = ipDF.crossJoin(ipDF)
# explain is fine, but actions would take a very long time.
cartesianDF2.explain()
# show / first can be ok as it only requires one partition, but count will be problematic.
cartesianDF2.first()

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's work through an example use-case. Our dataset contains a JSON file of IP address ranges allocated to various countries (`country_ip_ranges.json`), as well as various IP addresses of interesting transactions (`transaction_ips.json`). We want to work out the country code of a transaction's IP address. This can be accomplished using a ranged query to check if the transaction's address falls between one of the known ranges.
# MAGIC 
# MAGIC First, let's inspect our dataset.

# COMMAND ----------

ipRangesDF = spark.read.json("dbfs:/mnt/training/dataframes/country_ip_ranges.json")
transactionDF = spark.read.json("dbfs:/mnt/training/dataframes/transaction_ips.json")

ipRangesDF.show(5)
transactionDF.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC Next we want to run the query and check what range contains the transactions' IP addresses, and, thus, the country where the transaction occurred.

# COMMAND ----------

# TODO

ipByCountry = transactionDF.join(ipRangesDF, (transactionDF["ip_v4"] >= ipRangesDF["range_start"]) & (transactionDF["ip_v4"] <= ipRangesDF["range_end"]))
ipByCountry.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC Since we are expecting only the first partition using `show()` the result is rendered almost instantly. We can review what type of join resulted from the operation using the `explain()` function to render the physical plan.

# COMMAND ----------

ipByCountry.explain()

# COMMAND ----------

# MAGIC %md 
# MAGIC As expected, the resulting join was a `CartesianProduct`. One easy optimization is to pick the smaller of the two DataFrames and hint at broadcasting it, to instead achieve a `BroadcastNestedLoopJoin` instead.

# COMMAND ----------

# TODO

from pyspark.sql import functions as F

ipByCountryBroadcasted = transactionDF.join(F.broadcast(ipRangesDF), (transactionDF["ip_v4"] >= ipRangesDF["range_start"]) & (transactionDF["ip_v4"] <= ipRangesDF["range_end"]))
ipByCountryBroadcasted.show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's review the type of join resulting from the broadcast hint.

# COMMAND ----------

ipByCountryBroadcasted.explain()
