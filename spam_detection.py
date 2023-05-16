#!/usr/bin/env python
# coding: utf-8

import pyspark

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('spam').getOrCreate()

# Read the CSV file into a DataFrame
df = spark.read.csv("./Data/spam.csv", header=True, inferSchema=True)
df.printSchema()

# Drop unnecessary columns from the DataFrame
cols = ("_c2", "_c3", "_c4")
df = df.drop(*cols)
df.printSchema()

# Rename columns in the DataFrame
df = df.withColumnRenamed('v1', 'class').withColumnRenamed('v2', 'text')
df.show(3)

from pyspark.sql.functions import length

# Add a new column with the length of the 'class' column
df = df.withColumn('length', length(df['class']))
df.show(3)

# Group the DataFrame by the 'class' column and calculate the mean
df.groupBy('class').mean().show()

from pyspark.sql.functions import when, col

# Update the values in the 'class' column to 'ham' if they match specific conditions
df = df.withColumn('class', when(col('class').isin('ham"""', 'ham'), 'ham').otherwise(col('class')))
df.show()

# Remove rows with missing values
df = df.na.drop()

df.groupBy('class').mean().show()

from pyspark.ml.feature import (
    CountVectorizer, Tokenizer, StopWordsRemover, IDF, StringIndexer
)

# Apply tokenization, stop word removal, count vectorization, IDF, and label indexing to the text data
tokenizer = Tokenizer(inputCol='text', outputCol='token_text')
stop_remove = StopWordsRemover(inputCol='token_text', outputCol='stop_token')
count_vec = CountVectorizer(inputCol='stop_token', outputCol='c_vec')
idf = IDF(inputCol='c_vec', outputCol='tf_idf')
ham_spam_to_numeric = StringIndexer(inputCol='class', outputCol='label')

from pyspark.ml.feature import VectorAssembler

# Combine the TF-IDF and length columns into a single feature vector
clean_up = VectorAssembler(inputCols=['tf_idf', 'length'], outputCol='features')

from pyspark.ml.classification import NaiveBayes

# Create a Naive Bayes classifier
nb = NaiveBayes()

from pyspark.ml import Pipeline

# Create a pipeline to chain the preprocessing and classification stages
pipeline = Pipeline(stages=[ham_spam_to_numeric, tokenizer, stop_remove, count_vec, idf, clean_up])

# Fit the pipeline to the DataFrame and transform the data
cleaner = pipeline.fit(df)
clean_df = cleaner.transform(df)
clean_df = clean_df.select('label', 'features')
clean_df.show(3)

# Split the DataFrame into training and testing datasets
train, test = clean_df.randomSplit([0.7, 0.3])

# Fit the Naive Bayes classifier on the training data
spam_detector = nb.fit(train)

# Make predictions on the testing data
predictions = spam_detector.transform(test)
predictions.show(3)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Evaluate the accuracy of the predictions
evaluator = MulticlassClassificationEvaluator()
print('Test Accuracy: ' + str(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})))
