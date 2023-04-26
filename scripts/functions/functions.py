'''
Defining all functions needed for model construction.

by Saul Rincon
'''
# Importing modules needed
import os

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

# from matplotlib import pyplot as plt

import sklearn as sk

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error

from tqdm import tqdm


pd.options.mode.chained_assignment = None

path = './dia_analisis/scripts'

def excel_reader(path):
    """
    Reads excel file with raw data from stores, list of DataFrames
    with raw data to be transformed

    Args:
        path (_type_): string path to data file

    Yields:
        _type_: _description_
    """
    os.chdir('./scripts')
    # Gets the columns names and amount of rows per store for chunk reading the raw data.
    columns_to_use = np.array(pd.read_excel('raw_data.xlsx',skiprows=lambda x: x > 0, header=0, sheet_name='Hoja2').columns)
    stores_raw_df = pd.read_excel('raw_data.xlsx', usecols= ['BWTDA'], sheet_name='Hoja2')
    stores_rows = stores_raw_df.value_counts(sort=False).values
    stores_df = stores_raw_df.drop_duplicates()
    stores_df['rows'] = stores_rows
    #Loop that creates a generator object and finally yields DF store by store.
    row_counter = 0
    for index, row in stores_df.iterrows():
        raw_df = pd.read_excel('raw_data.xlsx', 
                                names=columns_to_use, 
                                usecols=columns_to_use,
                                skiprows=lambda x: not row_counter <= x <= (row_counter + row[1]), 
                                sheet_name='Hoja2')
        row_counter += row[1]
        yield raw_df, row[0]


def csv_generator(path):
    """
    Reads excel file with raw data from stores, list of DataFrames
    with raw data to be transformed

    Args:
        path (_type_): string path to data file

    Yields:
        _type_: _description_
    """
    os.chdir('./scripts')
    # Gets the columns names and amount of rows per store for chunk reading the raw data.
    raw_df = pd.read_excel('raw_data.xlsx', sheet_name='Hoja2')
    raw_df.drop(raw_df[raw_df['Area de Salida'] != 'Seco'].index, inplace = True)
    #Loop that creates a generator object and finally yields DF store by store.
    raw_df.replace("#N/D", np.NaN, regex=False, inplace=True)
    raw_df.dropna(inplace=True)
    raw_df['Dia sem'] = raw_df['Dia sem'].astype(str)
    # raw_df.reset_index()
    # raw_df.index.name = 'Index'
    # WRITE TO CSV
    # raw_df.to_csv('raw_data.csv')
    unique_suc_df = raw_df['BWTDA'].unique().astype(int)
    return raw_df, unique_suc_df


def dataframe_transformer(raw_df:pd.DataFrame, cross_val=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes raw dataframe, executes data cleansing and transformation. Adds
    weighted column for 'days'. Splits into two dataframes, one for training a
    statistical model, one for testing the model.

    Args:
        raw_df (pd.DataFrame): raw dataframe from a store
        suc (int): store's serial number

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, int]: 
                Dataframe: Training Pandas Dataframe
                Dataframe: Testing Pandas Datafram
                int: Store's serial number
    """
    # raw_df = raw_df[raw_df['Area de Salida'] == 'Seco']
    raw_df = raw_df[['Bultos', 'Dia sem', 'Contenedores']]
    raw_df.replace("#N/D", np.NaN, regex=False, inplace=True)
    raw_df.dropna(inplace=True)
    raw_df['Dia sem'] = raw_df['Dia sem'].astype(str)
    df_x = pd.get_dummies(raw_df[['Bultos','Dia sem']], dtype=int)
    df_y = raw_df[['Contenedores']]
    if cross_val:
        return df_x, df_y
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, train_size=.75) 
    return x_train, x_test, y_train, y_test


def model_constructor(x_trainset, y_trainset, power:int, cross_val=False)->LinearRegression:
    """Construct a statistic model based on trainig data and a given
    degree for the equation's power.

    Args:
        train_df (pd.DataFrame): Dataframe with training data.
        degree (int): Integer for determining the power of the model's equation.

    Returns:
        _type_: model equation
    """
    # corr = train_df[['Contenedores', '%Surtido']].corr('pearson')['%Surtido'][0]
    # Applying Polynomial Feature to X data
    if power == 1:
        x_data = x_trainset
    else:
        poly_feature = PolynomialFeatures(power)
        x_data = poly_feature.fit_transform(x_trainset)
    model = LinearRegression()
    if cross_val:
        model_error = cross_val_score(model, x_data, y_trainset, cv=3)
        return model_error
    model.fit(x_data, y_trainset)
    return model#, corr


def model_predictor(model, x_testset, y_testset, power):
    """Predicts the resulting data based on a statistic model and a given integer
    for the equation's power. Such power is used to give format to the X axis
    values to pass through the model. The data comes from a testing dataframe.
    It also compares the predicted data to the dataframes Y axis data and calculates
    the r squared value.

    Args:
        model (scikit Lear Linear Regression): Linear Regression model used to make the prediction.
        test_df (Pandas Dataframe): Dataframe with the data to pass through the model.
        degree (integer): represents the model's equation power. it
        is used to transform the X's values that are passed to the model for the prediction


    Returns:
        prediction_array (Array): Array with the predicted values.
        error(float): Represents r squared value.
    """
    if power == 1:
        x_data = x_testset
    else:
        poly_feature = PolynomialFeatures(power)
        x_data = poly_feature.fit_transform(x_testset)
    prediction_array = model.predict(x_data)#[:,0]
    prediction_array[prediction_array<0] = 0
    error = r2_score(y_testset, prediction_array)
    return prediction_array, error


def neural_predictor(power:int, df_x:pd.DataFrame, df_y:pd.DataFrame):
    X = df_x.values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_poly = np.concatenate([X**n for n in range(power)], axis=1) 
    y_array = df_y.values.reshape(-1, 1)
    if power == 1:
        epochs = 500
    else:
        epochs = 1200
    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_poly.shape[1],)),
        tf.keras.layers.Dense(32, activation='linear'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
        ])

    # Compile the model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Train the model on the input data and labels
    history = model.fit(X_poly, y_array, epochs=epochs, validation_split=.15)
    prediction = model.predict(X_poly).round()
    prediction[prediction<0] = 0
    error = r2_score(df_y,prediction)
    return error, prediction, model



def output_df_constructor():
    pass


def model_iterator(data_reader,
                   data_transformer,
                   model_train, 
                   model_predictor):
    # print(os.getcwd())
    # for raw_df, suc in data_reader(path):
    df, suc_array = csv_generator(path)
    for suc in suc_array:
        suc_df = df[df['BWTDA'] == suc]
        if len(suc_df.index) < 30:
            print(f'Sucursal {suc} no cuenta con suficientes datos para generar modelo')
            continue
        out_prediction = []
        out_error = 0
        out_power = 0
        for i in range(20):
            x_train, x_test, y_train, y_test = data_transformer(suc_df)
            for power in range(1,10):
                if power == 1:
                    model = model_train(x_train, y_train, power)
                    prediction, r2_error= model_predictor(model, x_test, y_test, power)
                    r2_diff = 1 - r2_error
                    used_power = power
                    used_model = model
                    if r2_error < 0:
                        r2_error = 0
                        r2_diff = 1
                        continue
                else:
                    model = model_train(x_train, y_train, power)
                    new_prediction, r2_new_error = model_predictor(model, x_test, y_test, power)
                    r2_new_diff = 1 - r2_new_error
                    if r2_new_error < 0:
                        continue
                    if r2_new_diff < r2_diff:
                        r2_error = r2_new_error
                        r2_diff = r2_new_diff
                        prediction = new_prediction
                        used_power = power
                        used_model = model
            if r2_error > out_error:
                out_error = r2_error
                out_prediction = prediction
                out_power = used_power
        print(out_prediction.round(0), 'r2 = ' + str(round(out_error, 2)), 'sucursal = ' + str(suc), 'power = ' + str(out_power))


def model_cross_val(data_transformer):
    cross_v = True
    df, suc_array = csv_generator(path)
    for suc in suc_array:
        suc_df = df[df['BWTDA'] == suc]
        if len(suc_df.index) < 30:
            print(f'Sucursal {suc} no cuenta con suficientes datos para generar modelo')
            continue
        out_error = 0
        out_power = 0
        x_df, y_df = data_transformer(suc_df, cross_val=cross_v)
        for power in range(1,10):
            if power == 1:
                r2_error_arr = model_constructor(x_df,y_df, power, cross_val=cross_v)
                r2_error = np.mean(r2_error_arr)
                r2_diff = 1 - r2_error
                used_power = power
                if r2_error < 0:
                    r2_error = 0
                    r2_diff = 1
                    continue
            else:
                r2_new_error_arr = model_constructor(x_df,y_df, power, cross_val=cross_v)
                r2_new_error = np.mean(r2_new_error_arr)
                r2_new_diff = 1 - r2_new_error
                if r2_new_error < 0:
                    continue
                if r2_new_diff < r2_diff:
                    r2_error = r2_new_error
                    r2_diff = r2_new_diff
                    used_power = power
        if r2_error > out_error:
            out_error = r2_error
            out_power = used_power
        print('r2 = ' + str(round(out_error, 2)), 'sucursal = ' + str(suc), 'power = ' + str(out_power))



def model_neural(data_transformer):
    cross_v = True
    # Reads data from excel file and build a Dataframe with the data, also an array of the stores
    df, suc_array = csv_generator(path)
    
    # Looping across the array of stores to build the models, one for each store.
    for suc in suc_array:
        suc_df = df[df['BWTDA'] == suc]

        # Stores with less than 30 data are not suitable for modeling
        if len(suc_df.index) < 30:
            print(f'Sucursal {suc} no cuenta con suficientes datos para generar modelo')
            continue
        if suc > 20:
            break
        #Initialazing output variables
        out_error = 0
        out_power = 0

        #Giving shape to the data before processing it.
        x_df, y_df = data_transformer(suc_df, cross_val=cross_v)
        print('Iniciando entrenamiento para sucursal: ' + str(suc))

        #Starting to loop across a range 1-5 for the regression degree, will keep the degree
        #with the best performance based on the r-squared.
        for power in tqdm(range(1,6), colour='green'):
            if power == 1:
                r2_error_arr, prediction, out_model = neural_predictor(power,x_df,y_df)
                r2_error = np.mean(r2_error_arr)
                r2_diff = 1 - r2_error
                used_power = power
                if r2_error < 0:
                    r2_error = 0
                    r2_diff = 1
                    continue
            else:
                r2_new_error_arr, new_prediction, new_model = neural_predictor(power,x_df,y_df)
                r2_new_error = np.mean(r2_new_error_arr)
                r2_new_diff = 1 - r2_new_error
                if r2_new_error < 0:
                    continue
                if r2_new_diff < r2_diff:
                    r2_error = r2_new_error
                    r2_diff = r2_new_diff
                    used_power = power
                    used_model = new_model
                    prediction = new_prediction
        if r2_error > out_error:
            out_error = r2_error
            out_power = used_power
            out_model = used_model
        
        out_model.save(f'./out_models/store_{suc}_d{out_power}_r{str(round(out_error, 2))[2:]}')
        print('r2 = ' + str(round(out_error, 2)), 'sucursal = ' + str(suc), 'power = ' + str(out_power))
        # plt.plot(x_df['Bultos'], y_df)
        # plt.subplot(prediction)

# print(model_iterator(excel_reader, dataframe_transformer, model_constructor, model_predictor))
tf.get_logger().setLevel('ERROR')
tf.keras.utils.disable_interactive_logging()
model_neural(dataframe_transformer)


# for raw_df, suc in excel_reader(path):    
#     print(raw_df.head())
#     break
# csv_generator(path)








