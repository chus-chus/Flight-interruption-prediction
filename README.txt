@author: Jesus Antonanzas Acero, Alex Carrillo Alza
@version: "1.0"
@email: "jesus.maria.antonanzas@est.fib.upc.edu, alex.carrillo.alza@est.fib.upc.edu"
@info: BDA, GCED, Big Data Analytics project
@date: 16/12/2019

-------------------------------------
Our project should contain 5 '.py' scripts:
- 'config.py': configures Spark environment when necessary
- 'load_into_hdfs.py': loads sensor csv's into HDFS in AVRO format
- 'data_management.py': creates training data matrix
- 'data_analysis.py': trains a decission tree classifier model on training data
- 'data_classifier.py': predicts new flight observations

These have to be put into the same directory as the 'resources' directory that
was included in he code skeleton, in order for the local reading option of the
sensor data (CSV's) to work.

'load_into_hdfs.py' is to be executed first if one wants to use HDFS. Note
that the HDFS path into which the files are to be loaded has to be explicitly
changed, as well as the reading path in 'data_management.py' and 'data_classifier.py'
if using this option.

If nothing is specified in 'data_management.py' or 'data_classifier.py', though,
CSV's will be read from local.

Then the order of execution is:
(optional): 'load_into_hdfs.py'
1. 'data_management.py'
2. 'data_analysis.py'
3. 'data_classifier.py'

Note that these scripts will write three objects:
1. 'data_matrix'
2. 'model'
3. 'test_matrix'
