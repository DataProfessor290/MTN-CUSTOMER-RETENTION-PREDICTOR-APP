def import_csv(url):
    import pandas as pd
    file = url
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    data = pd.read_csv(file)
    return data
df = import_csv("mtn_customer_churn.csv")


def config_data(data):
    
    """
    Configuring the columns in the dataframe
    """
    
    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.lower()
    return data

df = config_data(df)

def inspect(data):

    """
    Inspecting the data checking for the shape, structure, missing value, duplicated rows and number of unique values in each of the fatures on the dataframe in the dataset
    """
    
    print("checking the columns in the dataset")
    print()
    print(data.columns)

    print()
    print('The first five(5) rows in the dataset')
    print(data.head(5))

    print()
    print("The Number of rows and columns in the dataset")
    print(data.shape)
    print()

    print('The structure and datatype of the dataset')
   
    print()
    print(data.info())
    print()

    for value in data.isna().sum():
        if value > 1:
            print(f"There are {value} detected")
        else:
            print('There is no missing value detected')


    print()
    print(f"There are {data.duplicated().sum()} duplicated rows in the dataset")


inspect(df)


x = df.drop(["customer id", "full name", "reasons for churn", "customer churn status"], axis = 1)
y = df["customer churn status"].replace({
    "Yes" : 1,
    "No" : 0
})


def train(x,y):

    """
    Using a supervised approach to train and test the sets for modeling 
    """
    
    from sklearn.model_selection import train_test_split
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    resampled_x, resampled_y = RandomOverSampler().fit_resample(x, y)
    x_train,x_test,y_train,y_test = train_test_split(resampled_x,resampled_y,test_size=0.15, random_state=42)
    return x_train,x_test,y_train,y_test

    
x_train,x_test,y_train,y_test = train(x, y)


def process(data):

    """
    Processing and transforming the dataset for modeling 
    """
    
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer

    num_cols = data.select_dtypes(include=["number"]).columns
    cat_cols = data.select_dtypes(include=["object", "category"]).columns
	

   
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    processor = ColumnTransformer(
        transformers=[
            ('num_pipe', num_pipe, num_cols),
            ('cat_pipe', cat_pipe, cat_cols)
        ],
        remainder = 'passthrough'
    )
    return processor


transformer = process(x)
transformer



def train_xgboost(x_train, y_train, processor):

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, confusion_matrix
    import pandas as pd
    import matplotlib.pyplot as plt

    
    xgboost_model = Pipeline([
        ("transformer", processor),
        ("xgboost_model", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ]).fit(x_train, y_train)
    
    return xgboost_model


mtn_xgb_model = train_xgboost(x_train, y_train, transformer)

joblib.dump(mtn_xgb_model, "MTN XGB Classifier.pkl")