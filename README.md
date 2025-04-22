## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
sanjay kumar .B
212224230242
```
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/f2ce930a-fb66-4c51-a283-9c4e6cc2b73f)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/3ca97cd9-9637-4ab3-9484-f1afbb815c42)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/57fc56e4-e0ec-4402-9497-e04b476eec86)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/40cdf150-91c3-405f-8735-c8166ef44d05)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/572a5f11-9606-42f3-bec3-1245db698154)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/c84d3b73-a3b5-41dd-8f05-9e860394b268)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/cf5c2713-a0d3-4617-8228-1e5a2e5ff536)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/ccf3431e-0310-4522-887d-c51dfb530dfa)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/ab6697b5-e849-48f9-8e0d-3d172645f50f)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/c2f56b16-7c0a-4947-a223-9c74166bb9cd)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/6f5b6b86-55e8-414c-ada7-2ae779e38743)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/ef33eda8-d0a1-41c5-97de-39d14b17b14b)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/ec34f248-c828-46f0-9b2d-d0b72f95456f)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/d577cccd-aefc-4b53-9cb7-5e99233ef9cf)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/b7d2d028-c9f4-4c03-a854-b57454d7516a)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/8f645d34-dcd0-4cdf-a702-f1a2b74724a7)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/2f737125-9b94-4a84-ad9a-0da1e4be2811)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/73bdc368-84d1-41f9-b6fc-e947c8f4d486)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/be600bef-7072-4e6b-b45e-7e13baf9731b)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/2b14c245-9e22-42d3-9fc9-47b8d968d1f6)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/945b8b66-5bfa-468a-a5b2-044ad0d468f2)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c67266fb-0e79-48f5-a61d-80458965c572)
```
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
```
![image](https://github.com/user-attachments/assets/215d0864-8562-448c-a5ab-23a96e511723)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/11b078fc-a6ea-4390-9249-0762463d60a6)

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
