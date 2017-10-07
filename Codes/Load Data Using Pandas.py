# the most used python package to deal with data is called pandas we load it using
import pandas as pd  # and usually name it "pd"

df = pd.read_csv("DataSets\wheatData.data")  # create a data frame and assign the wheat data to it
# as a beginning its always great to check the data first, here are some examples :
print df  # view the whole data
print df.head(5)  # view first 5 records
print df.tail(5)  # view last 5 records

print df.describe  # view statistics about the data frame(min, max, standard deviation,...)
