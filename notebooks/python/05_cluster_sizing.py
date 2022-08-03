# Databricks notebook source
# MAGIC %md # Cluster Sizing
# MAGIC 
# MAGIC ## Lack of Memory
# MAGIC Partially cached DataFrames are considered an anti-pattern for Spark as Catalyst can no-longer efficiently optimize data processing. An easy way to avoid partial caches is to save data that doesn't fit in memory to disk. Another key factor is the location of the scratch directory, aka. the file system directory where Spark will persist shuffle data and DF partitions that do not fit into memory. The scratch directory is set to `/tmp` by default and configurable by:
# MAGIC - configuring the `spark.local.dir` setting at startup / runtime.
# MAGIC - editing spark's configuration file: `spark-env.sh`, export the `SPARK_LOCAL_DIRS` environment variable.

# COMMAND ----------

spark.conf.get("spark.local.dir", "")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Schema Effect on Memory
# MAGIC 
# MAGIC Tungsten's format for representing data is very lightweight compared to a Java object. But Tungsten will not decide the best data-type for a data point in our dataset. Thus, it's up to the developer to ensure that optimal types are used. Formatting data correctly can have a dramatic effect on how much memory the data takes up in Spark. For example, what if we stored a number in `String` form rather than an `Integer`?

# COMMAND ----------

intDF = spark.createDataFrame(map(lambda x: (x, ), range(1, (1024 * 1024 * 2) + 1)), ["num"])
stringDF = spark.createDataFrame(map(lambda x: (str(x), ), range(1, (1024 * 1024 * 2) + 1)), ["num_str"])

intDF.cache()
intDF.count()

stringDF.cache()
stringDF.count()

# COMMAND ----------

# MAGIC %md It's clear that representing the number uses much less memory.
# MAGIC 
# MAGIC <img src="https://github.com/rafaelvp-db/spark-tuning/blob/master/notebooks/img/cache_memory.jpg?raw=true" alt="Cached"/><br/>

# COMMAND ----------

intDF.unpersist()
stringDF.unpersist()

# COMMAND ----------

# MAGIC %md 
# MAGIC We can do even better for the above example. Assuming we're using large numbers, and we can't use an `Integer`, try to optimize the way we store the number further. 

# COMMAND ----------

stringDF2.unpersist()
longDF.unpersist()

# COMMAND ----------

# TODO

# This exercise can take awhile to run. Feel free to reduce the limit.
limit = 1024 * 1024 * 2

# let's create a DF full of numbers as a String
stringDF2 = spark.createDataFrame(map(lambda x: (x, ), ["2147483648"] * limit), ["string"])
# convert the below number to a long instead.
longDF = spark.createDataFrame(map(lambda x: (x, ), ["2147483648"] * limit), ["long"])

# now compare size of both DataFrames by caching and performing an action.

stringDF2.cache()
longDF.cache()

stringDF2.count()
longDF.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC Not quite as big a reduction in DF size as before. As `2147483648L` is larger than `Integer.MAX_VALUE`, but not that large a number, storing it as a long doesn't have the same dramatic effect as compared to when we used an `Integer`, but it still does reduce the DF size. There is another important point to consider if the number is stored as a String. There will be **casting** overhead at runtime that will slow down processing.
# MAGIC 
# MAGIC Remember, typically `Longs` are 64 bits in the JVM and `Ints` are 32 bits, while `Strings` are of variable length; so, for a DataFrame's size to be reduced, the String representation of the number needs to be larger than 64 bits (though be sure to account for strings being padded to fill out a word).

# COMMAND ----------

# remember to lean up any DFs we wont use going forward.
stringDF2.unpersist()
longDF.unpersist()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Hardware Provisioning
# MAGIC 
# MAGIC ### Memory
# MAGIC 
# MAGIC - Overallocating too much memory to a single JVM can be detrimental, as GC becomes more expensive over large heaps. Typically, the upper limit is approximately 40GB, but this depends on the workload and JVM configuration. 
# MAGIC - Spark documentation recommends allocating 75% of a machine's memory to Spark and reserving 25% for operating system etc. However, that rule of thumb
# MAGIC   doesn't always apply. Consider a machine with 256GB of physical RAM running a version of Linux. Does the operating system _really_ need 64GB of memory?
# MAGIC   No. In that kind of hardware configuration, you first want to figure out how much RAM the operating system requires to work efficiently. (In some cases,
# MAGIC   it might be less than 1GB.) Then, you can allocate the remaining RAM to Spark. (In this particular example, you'd likely want to run multiple Executors.)
# MAGIC - Powerful high-end machines with more than 200GB of memory should run multiple executors rather than a single large JVM.
# MAGIC 
# MAGIC **How much memory does my application need?** Benchmarking and testing yields the most accurate answer, but rough estimations can be made by loading a fraction of the dataset the application will process, caching it and using Spark's UI to work out memory used by Spark. 
# MAGIC 
# MAGIC Below is a DataFrame created from a CSV file of individuals. On disk this file is of size 289.4 MB. When cached in Spark, the file's size will change. How much memory will we require to cache 50% of this file?
# MAGIC 
# MAGIC **Note**: We're using an explicit schema, to speed things up.

# COMMAND ----------

# MAGIC %fs ls dbfs:/mnt/training/dataframes/people-with-header-5m.txt

# COMMAND ----------

# TODO
# Use Spark transformations and actions to work out how much memory is required to store 50% of this file.
# StorageLevel used greatly effects memory usage, so stick to the default StorageLevel.

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
peopleDF = (
  spark
    .read
    .schema(people_schema)
    .option("header", "true")
    .option("delimiter", ":")
    .csv("dbfs:/mnt/training/dataframes/people-with-header-5m.txt")
)
peopleHalfDF = peopleDF.sample(fraction = 0.5)
peopleHalfDF.cache()
peopleHalfDF.count()

# verify file size using the spark UI

# COMMAND ----------

# MAGIC %md 
# MAGIC If we directly assumed the file size of 289.4 MB to represent the dataset out cluster sizing estimation wouldn’t be as accurate as using tools available to us through spark. Caching the entire DataFrame would yield `229.5 MB`, significantly less than expected. This is due to Tungsten having an efficient storage format. 
# MAGIC 
# MAGIC Many other things need to be accounted for when choosing how much memory to allocate for an application:
# MAGIC - What kind of shuffling will occur
# MAGIC - Broadcast variables, broadcasting thresholds
# MAGIC - User memory - storing variables that will be serialized to Spark's executors. 

# COMMAND ----------

peopleHalfDF.unpersist()
# The 289.4 MB file requires about 229 MB of Spark memory to cache.
peopleDF.cache()
peopleDF.count()

# COMMAND ----------

# MAGIC %md Remember to unpersist any DataFrames that won't be used.

# COMMAND ----------

peopleDF.unpersist()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Network
# MAGIC 
# MAGIC Once an application's dataset fits in memory, the application typically becomes network-bound, especially for **distributed reduce** processing such as `groupBys`, `joins` and transformations requiring a shuffle. 
# MAGIC 
# MAGIC The simplest way to understand the impact of large shuffles on your application's performance is to use the Spark UI (available on `http://<driver-node>:4040` by default) to look at the volume of data shuffled and correlate it to how long it takes a stage to complete its processing. 

# COMMAND ----------

# MAGIC %md ### CPU Cores
# MAGIC 
# MAGIC As there is minimal sharing of resources between threads, Spark scales well with tens of CPU cores per machine. Recommendations for per-executor provisioning advise that a minimum of 8 Spark cores is a good starting point. Depending on how CPU intensive the workload of an application might be, more cores can be provisioned with the upper limit per executor being 16 cores. 
# MAGIC 
# MAGIC Again, these are guidelines, depending on the workload more cores might be required as typically once an application's data is in memory, that application becomes CPU or network bound. As before, the best way of reviewing available resources is to look at the Spark UI. This time, rather than looking at the main (driver) UI, we're the Spark Master UI, which is typically available on `http://<spark-master-node>:8080` (for vanilla Spark distributions) and in DataBricks under `Clusters > Spark UI > Spark Cluster UI - Master`.
# MAGIC 
# MAGIC <img src="https://github.com/rafaelvp-db/spark-tuning/blob/master/notebooks/img/spark_master.png?raw=true"/>

# COMMAND ----------

# MAGIC %md ## Single Application vs Multi User Environment
# MAGIC 
# MAGIC Within a Spark application, multiple jobs can be executed by Spark's scheduler. Since the scheduler is fully thread-safe, separate threads can safely start jobs in parallel. This is a highly desirable feature in a multi-user environment with an application that is trying to get a response quickly and seems very responsive.
# MAGIC 
# MAGIC 
# MAGIC ##### FIFO scheduler
# MAGIC - By default, Spark's scheduler runs jobs in a first-in-first-out approach (FIFO). 
# MAGIC - Each job is processed by Spark's scheduler and broken down it to a job of stages of tasks. 
# MAGIC - After scheduling, the job is submitted for execution. 
# MAGIC - If it's the first job at the head of the queue, it will greedily acquire all available resources. This is fine if it's a small job, but should it require all the resources, other jobs in the queue can get delayed significantly.
# MAGIC 
# MAGIC ##### Fair Scheduler
# MAGIC - After Spark 0.8, a fair scheduler was introduced to allow for sharing of resources between jobs.
# MAGIC - In the fair scheduler, Spark uses a round-robin assignment of resources so that all jobs requiring resources get an equal share.
# MAGIC - This allows short jobs that are submitted after large resource heavy jobs to start receiving resources right away, improving response times.
# MAGIC - The fair scheduler is best suited to multi-user environments.
# MAGIC 
# MAGIC You can verify which scheduler is being used through the Spark session:

# COMMAND ----------

spark.conf.get("spark.scheduler.mode")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Cluster Sizing Case Study
# MAGIC 
# MAGIC Assuming we've got a dataset of roughly 60 GB we need to process. The below cluster should be sufficient. 
# MAGIC 
# MAGIC Total Cluster Memory: **95 GB** <br/>
# MAGIC Total Cores: **52 Cores** <br/>
# MAGIC 
# MAGIC ## Reasoning
# MAGIC 
# MAGIC Let's break this down. Why do we need 95 GB of Spark to process a 60 GB dataset? 
# MAGIC 
# MAGIC 75% of all executor memory is available for storing the dataset. So 75% of 95 GB is **71.25 GB**. The availability of all that memory depends on what kind of aggregation buffers get created. Only **50%** of that 71.25 GB is reserved for our DataFrames and DataSets (36.6 GB). If our dataset spills beyond the `spark.memory.storageFraction` area (which is immune to cache eviction), that part of our dataset that "spills over" might get garbage collected and released from memory to make room for a shuffle occurring between stages.
# MAGIC 
# MAGIC If this is the case, we will incur a potentially expensive re-computation. Assuming there are not many large shuffles in the processing that we are doing for our 60 GB dataset, the cluster size should be sufficient. 
# MAGIC 
# MAGIC ### CPU Intensive Workloads
# MAGIC 
# MAGIC  - The number of (Spark) cores required depends on how compute-heavy the work load turns out to be. 
# MAGIC  - The more CPUs the cluster has access to, the faster jobs will perform.
# MAGIC  - Typically once the dataset fits in memory and is loaded, CPU and network are the bottleneck for an application.
# MAGIC  - This isn't always the case. Some applications exhaust CPU resources before running out of memory, such as ML applications and applications where only part of the dataset is loaded. 
# MAGIC 
# MAGIC 
# MAGIC ### Minimal configuration to build the above cluster
# MAGIC 
# MAGIC For each of the 3 machines we can edit `spark-env.sh` in order to build the above cluster:
# MAGIC 
# MAGIC Machine 1:
# MAGIC ```
# MAGIC export SPARK_EXECUTOR_INSTANCES=1
# MAGIC export SPARK_EXECUTOR_CORES=12
# MAGIC export SPARK_EXECUTOR_MEMORY=23g
# MAGIC export SPARK_DAEMON_MEMORY=1g
# MAGIC ```
# MAGIC 
# MAGIC Machine 2:
# MAGIC ```
# MAGIC export SPARK_EXECUTOR_INSTANCES=1
# MAGIC export SPARK_EXECUTOR_CORES=8
# MAGIC export SPARK_EXECUTOR_MEMORY=24g
# MAGIC ```
# MAGIC 
# MAGIC Machine 3:
# MAGIC ```
# MAGIC export SPARK_EXECUTOR_INSTANCES=2
# MAGIC export SPARK_EXECUTOR_CORES=16
# MAGIC export SPARK_EXECUTOR_MEMORY=24g
# MAGIC ```

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Future-proof Sizing
# MAGIC 
# MAGIC The setup shown above might be enough to handle the requirements of the initial application; but, as time progresses and a business grows, so, too, will the data that it's gathering. Spark was designed for commodity hardware, so that users can to start with small clusters and scale up as requirements change. 
# MAGIC 
# MAGIC **Projecting** expected growth over time as the business grows can allow application developers to scale their resources before the cluster they're using slows down or hits memory problems.
# MAGIC 
# MAGIC With the popularity of **cloud** deployments this has become even simpler, as users don't need to buy more hardware and add it to a cluster; instead, they can rent additional resources as needed.
