# Databricks notebook source
# MAGIC %md # Memory Usage
# MAGIC 
# MAGIC Determining memory usage in Spark
# MAGIC - Logs - Many JVM options can be enabled to allow for debugging spark memory space.
# MAGIC - Spark UI - The storage tab can be very useful when it comes to working out memory utilization in spark. The executors tab shows details of memory usage as spark progresses through jobs.
# MAGIC 
# MAGIC The below code **will exhaust the 2GB memory** available to our CE Cluster. Running it can take a long time since disk access is required for the data that is being shuffled. 

# COMMAND ----------

names = spark.read.parquet("dbfs:/mnt/training/ssn/names.parquet")
people2 = (
  spark
    .read
    .option("inferSchema", "true")
    .csv("dbfs:/mnt/training/dataframes/people-alt.csv")
    .toDF("firstName", "lastName", "gender", "DOB", "id", "state")
)
joinedDF = people2.join(names, names["firstName"] == people2["firstName"])
distDF = joinedDF.distinct().cache()
distDF.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC #####Spark UI
# MAGIC 
# MAGIC Let's review the executor and storage tabs in the Spark UI after the jobs completed. 
# MAGIC 
# MAGIC Executor tab:
# MAGIC - Good for an overview of how much data is being transferred between stages (shuffle)
# MAGIC - Summary of memory currently used in executor
# MAGIC - Shows time spent GCing
# MAGIC 
# MAGIC <img src="https://github.com/rafaelvp-db/spark-tuning/blob/master/notebooks/img/executors.png?raw=true" alt="Executor tab"/><br/><br/>    
# MAGIC 
# MAGIC Storage tab:
# MAGIC - A lot more details about how data is cached
# MAGIC - Fraction of partitions cached / spilled to disk.
# MAGIC 
# MAGIC <img src="https://github.com/rafaelvp-db/spark-tuning/blob/master/notebooks/img/storage.png?raw=true" alt="Storage tab"/><br/><br/>    
# MAGIC 
# MAGIC #####Logs
# MAGIC 
# MAGIC The spark UI is a great resource thanks to the visualizations, but the logs contain the finely grained details of how memory is being utilized.  We can see the exact values of what was processed and how, down to the individual bytes.
# MAGIC ```
# MAGIC 17/06/14 08:56:28 INFO TaskMemoryManager: 624951296 bytes of memory are used for execution and 1391857080 bytes of memory are used for storage
# MAGIC 17/06/14 08:56:43 INFO TaskMemoryManager: 654311424 bytes of memory are used for execution and 1389232456 bytes of memory are used for storage
# MAGIC ```
# MAGIC The logs also allow us to create a time-line of how an RDD / DF was cached and what memory pressure each partition introduced to the cluster.
# MAGIC ```
# MAGIC 17/06/14 08:57:01 INFO BlockManagerInfo: Added rdd_18_150 in memory on 10.172.236.84:38470 (size: 13.8 MB, free: 702.4 MB)
# MAGIC 17/06/14 08:57:38 INFO BlockManagerInfo: Added rdd_18_169 in memory on 10.172.236.84:38470 (size: 13.7 MB, free: 618.6 MB)
# MAGIC 17/06/14 08:58:15 INFO BlockManagerInfo: Added rdd_18_195 in memory on 10.172.236.84:38470 (size: 14.2 MB, free: 659.6 MB)
# MAGIC ...
# MAGIC 17/06/14 08:56:43 INFO BlockManagerInfo: Added rdd_18_51 on disk on 10.172.236.84:38470 (size: 10.4 MB)
# MAGIC 17/06/14 08:56:43 INFO BlockManagerInfo: Added rdd_18_52 on disk on 10.172.236.84:38470 (size: 9.8 MB)
# MAGIC ```

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Tungsten
# MAGIC 
# MAGIC Tungsten is the in-memory storage format for **Spark SQL / DataFrames**. Advantages:
# MAGIC 
# MAGIC - Compactness: Column values are encoded using custom encoders, not as JVM objects (as with RDDs). The benefit of using Spark 2.x's custom encoders is that you get almost the same compactness as Java serialization, but significantly faster encoding/decoding speeds. Also, for custom data types, it is possible to write custom encoders from scratch.
# MAGIC - Efficiency: Spark can operate directly out of Tungsten, without deserializing Tungsten data into JVM objects first. 
# MAGIC 
# MAGIC Let's see an example of how Tungsten can impact the size of a cached dataset. 
# MAGIC 
# MAGIC More information on the <a href="https://databricks.com/blog/2015/04/28/project-tungsten-bringing-spark-closer-to-bare-metal.html" target = '_blank'>Tungsten project</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Java String Memory Allocation
# MAGIC 
# MAGIC <br/>
# MAGIC 
# MAGIC - A regular 4 byte string would end up taking 48 bytes. 
# MAGIC - The diagram shows how the 40 bytes are allocated and we also need to round up byte usage to be divisible of 8 due to JVM padding. 
# MAGIC - This is a very bloated representation knowing that of these 48 bytes, we're actually after only 4. 

# COMMAND ----------

# If you did run the cell above, you'll want to free the cached DataFrame.
distDF.unpersist()

# COMMAND ----------

# Demo: Tungsten reducing space complexity.
data = range(1, 5 * 1024 * 1024 + 1)

rdd1 = spark.sparkContext.parallelize(data)
rdd1.cache()
rdd1.first()

df1 = rdd1.map(lambda x: (x,)).toDF(["Number"])
df1.cache()
df1.first()

# COMMAND ----------

# MAGIC %md 
# MAGIC Note:
# MAGIC 
# MAGIC - The entire dataset was not cached; only a portion of it was cached. Since the `first` action was used, only one of the partitions was required to satisfy the
# MAGIC   action. Only that partition was read from the file, and only that partition was cached.
# MAGIC - When stored under Tungsten's format, the dataset takes up approximately 1/4 of the space compared to when it's stored as a Java object. 
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/lyubent/i/master/tungsten-vs-javaobj.png" alt="Storage tab"/><br/><br/>    

# COMMAND ----------

# MAGIC %md 
# MAGIC The below cell illustrates that 100 / num_partitions yields the fraction cached when the first() action is used. It depends on `spark.default.parallelism`, which can differ per cluster. 

# COMMAND ----------

print("spark.default.parallelism: " + str(spark.sparkContext.defaultParallelism))
print(100.0 / df1.rdd.getNumPartitions())

# COMMAND ----------

# clean up
rdd1.unpersist()
df1.unpersist()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Partially Cached DataFrame or Dataset
# MAGIC 
# MAGIC - Considered an anti-pattern
# MAGIC - Default `cache()` = `persist(StorageLevel.MEMORY_AND_DISK_SER)` to avoid DF re-computation
# MAGIC - Better to read from DISK than to recompute
# MAGIC - With RDDs you could only alter a storage level once or an exception would be thrown
# MAGIC - With DFs, changing levels is fine. However, the change in storage will not overwrite the first value.

# COMMAND ----------

from pyspark import StorageLevel

df2 = spark.createDataFrame(map(lambda x: (x, ), range(1, 51))).withColumnRenamed("_1", "numb")

df2.persist(StorageLevel.DISK_ONLY)
# cache only first partition in DF
df2.first()
# change storage level
df2.persist(StorageLevel.MEMORY_ONLY)
# force full cache of DF
df2.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Storage
# MAGIC 
# MAGIC | RDD Name                  | Storage Level                 | Cached Partitions | Fraction Cached | Size in Memory | Size on Disk |
# MAGIC |---------------------------|-------------------------------|-------------------|-----------------|----------------|--------------|
# MAGIC | LocalTableScan [numb#268] | Disk Serialized 1x Replicated |                 8 |            100% |          0.0 B |       3.9 KB |

# COMMAND ----------

# MAGIC %md ## GC tuning
# MAGIC 
# MAGIC Picking a suitable garbage collector based on how spark is used plays a major part in performance. There are many choices and defaults can change depending on the environment used. 
# MAGIC 
# MAGIC ### Heap Space for Parallel and CMS
# MAGIC The idea of the parallel and CMS GC algorithms is that they can carry out two types of garbage collection. A young generation GC and an old generation GC in an attempt to identify how long an object has been alive for. As the age of an object increases, it's likelihood to run through the check for garbage collection decreases. This is not to say that a very old object won't be collected if it's no longer referenced, it will just take the JVM longer to identify such an object for collection.
# MAGIC 
# MAGIC <img src="https://www.journaldev.com/wp-content/uploads/2014/05/Java-Memory-Model.png" alt="JVM Heap Space"/><br/>
# MAGIC - Newly allocated objects are placed in the Eden space
# MAGIC - Once the Eden space is full, this is likely to trigger a Young Gen garbage collection. 
# MAGIC - Any Eden objects that survive will be promoted to the survivor space.
# MAGIC - Objects in the survivor space are moved between S0 and S1 each time young gen GC runs, as they are copied they are also aged.
# MAGIC - Once survivor objects reach an aging threshold, they are promoted to the old gen space. 
# MAGIC - Old Gen contains objects that have survived several Young Gen GCs
# MAGIC - **Young Generation Garbage Collection** - Searches through the Eden and Survivor spaces, to GC objects that are no longer referenced. Any objects that remain as alive, can potentially be promoted to the old generation space. 
# MAGIC - **Old Generation Garbage Collection** - Searches through the Old Gen space to find objects that can be collected. 
# MAGIC 
# MAGIC 
# MAGIC ### Parallel GC
# MAGIC 
# MAGIC In the Parallel GC algorithm garbage collection can use multiple threads to collect unreferenced objects. The Parallel GC algorithm prioritizes low pause times and is well suited to batch applications. Real-time applications would suffer greatly from this type of GC as it pauses application threads.
# MAGIC 
# MAGIC <img src="https://www.techgeeknext.com/img/garbage-collector/garbage-collector-types.PNG" alt="Parallel GC"/><br/> 
# MAGIC <br/>
# MAGIC - An algorithm that has stop-the-world garbage collection events. Both young and old GCs pause application threads. 
# MAGIC - Goals of Parallel GC are addressed in the following order:
# MAGIC     1. Maximum pause time goal - Application threads shouldn't be paused longer than `XX:MaxGCPauseMillis`
# MAGIC     2. Throughput goal - A ration of time spent running application threads to time spent garbage collecting. Specified via `XX:GCTimeRatio` setting. Default is 1/99 meaning 1% of total run time is spent in garbage collection.  
# MAGIC     3. Minimum footprint goal - Keep memory used by application as low as possible. 
# MAGIC     
# MAGIC <br/>
# MAGIC 
# MAGIC ### CMS 
# MAGIC <br/>   
# MAGIC - Uses multiple threads concurrently with application to scan for unreferenced objects
# MAGIC - CSM encounters Stop the world events in two scenarios:
# MAGIC     1. Initial mark - During the initial heap scan for root objects (objects in old gen that are reachable by threads entry points or static variables)
# MAGIC     2. Remark - When the application has changed heap state as CMS was running concurrently. CMS now has to go back and carry out a stop the world remark to verify it will collect the desired objects.
# MAGIC - CMS requires more CPU but allows the application threads more continuous execution time without pausing them
# MAGIC - Use cases are typically applications that can not afford to have large pause times.
# MAGIC 
# MAGIC <br/>
# MAGIC 
# MAGIC ### G1
# MAGIC <br/>
# MAGIC <img src="https://www.techgeeknext.com/img/garbage-collector/garbage-collector-types.PNG"/><br/>
# MAGIC <br/>
# MAGIC - Heap split into regions (typically 2MB, configurable via the `XX:G1HeapRegionSize` JVM flag.)
# MAGIC - GC Pauses become more predictable
# MAGIC - Regions assigned as Eden, Survivor, Old Gen or Humongous*
# MAGIC - Humongous regions are old gen regions, that are allocated contiguously by JVM. 
# MAGIC     - An allocation is considered "humongous" for objects larger than 50% of a region in G1.
# MAGIC     - These regions are used for large objects.
# MAGIC - In old gen GC, collecting regions with 0 live objects is easy
# MAGIC     - Rarely the case.
# MAGIC     - Old Gen GC works out *best regions* to GC.
# MAGIC     
# MAGIC #### G1 Considerations
# MAGIC 
# MAGIC - Avoiding full GC - Full GC in G1 causes application threads to be paused so the garbage collecting algorithm can identify objects that are unreferenced and thus need to be garbage collected:
# MAGIC     1. We can reduce frequency of full GCs by reducing `InitiatingHeapOccupancyPercent` which by default is configured to 45. Reducing this setting allows G1 to start initial concurrent marking at an earlier time. 
# MAGIC     2. Increasing `ConcGCThreads` will allow G1 to utilize more threads for concurrent marking. The tradeoff here is that this will take away CPU time from application threads and give them to GC. 
# MAGIC - Region size - G1 will decide region size depending on the heap size used. Region size can have significant performance implications as it dictates when an object is considered to be "humongous" and would require contiguous region allocation. Region size can be set via the `-XX: G1HeapRegionSize=<size>` JVM option.
# MAGIC - Avoiding Humongous allocation - Allocation of large objects takes longer than the regular process, especially with fragmented heaps. In Java 8u40 <a href="https://bugs.openjdk.java.net/browse/JDK-8027959" target = '_blank'>significant updates</a>
# MAGIC were added to optimize reclaiming heap space by GCing such objects, but overall collecting humongous objects is still relatively expensive. 
