import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("../input/datanew/heart_failure.csv")
data.head(5)
data.describe()
data.isnull().sum()
data['DEATH_EVENT'].value_counts()
data.loc[data['age'] <= 60, 'Age_category'] = 'Before_retire' 
data.loc[data['age'] > 60, 'Age_category'] = 'After_retire' 
print (data)
data.head()
data['Age_category'].value_counts()
           
           
data_new= data.drop('Age_category',axis=1)
x =  data_new.drop('DEATH_EVENT',axis=1)

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x
        
add_scaler = MinMaxScaler()
x = pd.DataFrame(x).reset_index(drop=True).infer_objects()
cat_train_x, num_train_x = catch_num(x)
add_scaler.fit(num_train_x)
num_train_x = pd.DataFrame(add_scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
x = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)
        
y= data_new['DEATH_EVENT']
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
x_sTrain, x_sTest, y_Train, y_Test = train_test_split(x, y, train_size=0.8, test_size=1-0.8, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors=1)
#knn.fit(x_sTrain,y_Train)
#predict_2 = knn.predict(x_sTest)




import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
#print("start running model training........")
model = KNeighborsClassifier()
model.fit(x_sTrain, y_Train)
y_pred = model.predict(x_sTest)
score = accuracy_score(y_Test, y_pred)
import numpy as np
np.save("HybridPipeGen/core/tmpdata/prenotebook_res/prakashbhatt1386_eda-logistic-reg-decision-tree-knn.npy", { "accuracy_score": score })

