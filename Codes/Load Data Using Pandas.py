# the most used python package to deal with data is called pandas we load it using
import pandas as pd  # and usually name it "pd"
import matplotlib.pyplot as plt

df = pd.read_csv("DataSets\wheatData.data")  # create a data frame and assign the wheat data to it
# as a beginning its always great to check the data first, here are some examples :
df  # view the whole data
df.head(5)  # view first 5 records
df.tail(5)  # view last 5 records

#Statistics about the dataFrame
df.describe  # view statistics about the data frame(min, max, standard deviation,...)
df.hist() # creates multiple plots correspnds with the number of featues in the dataframe (df)

#Slicing and dicing dataframe
df.loc[[1],:] #show first row of the dataframe
df.loc[:,[1,2]] # show columns 1 & 2

