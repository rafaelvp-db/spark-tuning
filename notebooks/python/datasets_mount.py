# Databricks notebook source
# MAGIC %md ## Prepare data mounting for each lab

# COMMAND ----------

ACCESSY_KEY_ID = "XXXXXX"
SECERET_ACCESS_KEY = "XXXXX" 

mounts_list = [
{'bucket':'databricks-corp-training/common', 'mount_folder':'/mnt/training'},
{'bucket':'db-wikipedia-readonly-eu', 'mount_folder':'/mnt/wikipedia-readonly-eu'},
{'bucket':'db-wikipedia-readonly-eu', 'mount_folder':'/mnt/wikipedia'},
{'bucket':'databricks-corp-training/ml-amsterdam/mooc', 'mount_folder':'/mnt/spark-mooc'},
{'bucket':'databricks-corp-training/ml-amsterdam', 'mount_folder':'/mnt/ml-class'},
{'bucket':'db-wikipedia-readonly-use', 'mount_folder':'/mnt/wikipedia-readonly/'},
]

# COMMAND ----------

for mount_point in mounts_list:
  bucket = mount_point['bucket']
  mount_folder = mount_point['mount_folder']
  try:
    dbutils.fs.ls(mount_folder)
    dbutils.fs.unmount(mount_folder)
  except:
    pass
  finally: #If MOUNT_FOLDER does not exist
    dbutils.fs.mount("s3a://"+ ACCESSY_KEY_ID + ":" + SECERET_ACCESS_KEY + "@" + bucket,mount_folder)

# COMMAND ----------

display(dbutils.fs.mounts())
