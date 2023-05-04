import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# from main import VERSION, PREVIOUS_VERSION
VERSION = '0.0.2'
PREVIOUS_VERSION = '0.0.1'


class Model():
    def Process_Data(self):
        self.df = pd.read_csv('t3.csv')
        self.df['sex'].replace(['male', 'female', 'NO'],
                               [1, 2, 3], inplace=True)
        self.df['native_sex'].replace(
            ['male', 'female', 'NO'], [1, 2, 3], inplace=True)

        self.df['side'].replace(['villain', 'hero', 'NO'], [
                                1, 2, 3], inplace=True)
        self.df['native_side'].replace(
            ['villain', 'hero', 'NO'], [1, 2, 3], inplace=True)

        self.df['type'].replace(['universal', 'blast', 'speed', 'combat'], [
            1, 2, 3, 4], inplace=True)
        self.df['native_type'].replace(['universal', 'blast', 'speed', 'combat'], [
            1, 2, 3, 4], inplace=True)

        self.Y_value = self.df[['1st_gear',
                                '2nd_gear', '3rd_gear', '4th_gear']].values

        self.X_value = self.df[
            ['sex', 'native_sex', 'type', 'native_type', 'side', 'native_side', 'native_tier', 'target_tier', 'is_premium',
             'is_extra_cost']].values

    def Fit_Data(self):
        self.model.fit(self.X_value, self.Y_value)
        self.Save_Model()

    def Save_Model(self):
        joblib.dump(self.model, f'mffgc_v{VERSION}.mffgc')

    def __init__(self) -> None:
        try:
            try:
                self.model = joblib.load(f'mffgc_v{VERSION}.mffgc')
            except:
                self.model = joblib.load(f'mffgc_v{PREVIOUS_VERSION}.mffgc')
        except:
            self.model = LinearRegression()
        self.Process_Data()
        self.Fit_Data()

    def Predict(self, predictData):
        return self.model.predict(np.array(
            [
                predictData['sex'],
                predictData['native_sex'],
                predictData['type'],
                predictData['native_type'],
                predictData['side'],
                predictData['native_side'],
                predictData['native_tier'],
                predictData['target_tier'],
                predictData['is_premium'],
                predictData['is_extra_cost']
            ]).reshape(1, -1))
