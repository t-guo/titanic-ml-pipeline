import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer


class CustomizableTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.state_dependent_transforms = {}

    def fit(self, data, target=None):
        pass


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col
        self.lb = None

    def fit(self, data):
        if self.lb is None:
            self.lb = LabelBinarizer()
            self.lb = self.lb.fit(data[self.target_col])

    def transform(self, data):
        output = data.copy()

        if data[self.target_col].dtype == np.float64 or data[self.target_col].dtype == np.int64:
            data[self.target_col] = data[self.target_col].astype(int).astype(str)

        x = self.lb.transform(data[self.target_col])

        encoded_df = pd.DataFrame(x)
        encoded_df.columns = [self.target_col + "_" + str(c).lower() for c in self.lb.classes_]

        output = pd.merge(output, encoded_df, left_index=True, right_index=True)
        output.drop(columns=self.target_col, inplace=True)

        return output


class NAImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, method="mean", target_value=None):
        self.target_col = target_col
        self.target_value = target_value
        self.method = method

    def fit(self, data):
        if self.target_value is None:
            if self.method == "mean":
                self.target_value = data[self.target_col].mean()
            elif self.method == "median":
                self.target_value = data[self.target_col].median()
            elif self.method == "mode":
                self.target_value = data[self.target_col].mode()
            else:
                raise NotImplementedError("Method not implemented, must be one of ['mean', 'mode', 'static']")

    def transform(self, data):
        output = data.copy()
        output[self.target_col] = output[self.target_col].fillna(self.target_value)

        return output


class TitanicFeatureTransformer(CustomizableTransformer):

    def transform(self, data):

        output = data.copy()

        # Rename columns to lower_case
        output.columns = [x.lower() for x in output.columns]

        # Extract title
        output["name"] = output["name"].fillna("")
        output["title"] = output["name"].apply(lambda x: self.get_title(x))

        # Extract gender
        output["male"] = 0
        output.loc[output["sex"] == "male", "male"] = 1

        # Extract and impute cabin info
        output["deck"] = output["cabin"].str.slice(0, 1)
        output["room"] = output["cabin"].str.slice(1, 5).str.extract("([0-9]+)", expand=False).astype("float")

        # Impute variables
        if "age_imputer" not in self.state_dependent_transforms.keys():
            age_imputer = NAImputer("age", method="median")
            age_imputer.fit(output)
            output = age_imputer.transform(output)
            self.state_dependent_transforms["age_imputer"] = age_imputer
        else:
            age_imputer = self.state_dependent_transforms["age_imputer"]
            output = age_imputer.transform(output)

        if "fare_imputer" not in self.state_dependent_transforms.keys():
            fare_imputer = NAImputer("fare")
            fare_imputer.fit(output)
            output = fare_imputer.transform(output)
            self.state_dependent_transforms["fare_imputer"] = fare_imputer
        else:
            fare_imputer = self.state_dependent_transforms["fare_imputer"]
            output = fare_imputer.transform(output)

        if "sibsp_imputer" not in self.state_dependent_transforms.keys():
            sibsp_imputer = NAImputer("sibsp")
            sibsp_imputer.fit(output)
            output = sibsp_imputer.transform(output)
            self.state_dependent_transforms["sibsp_imputer"] = sibsp_imputer
        else:
            sibsp_imputer = self.state_dependent_transforms["sibsp_imputer"]
            output = sibsp_imputer.transform(output)

        if "parch_imputer" not in self.state_dependent_transforms.keys():
            parch_imputer = NAImputer("parch")
            parch_imputer.fit(output)
            output = parch_imputer.transform(output)
            self.state_dependent_transforms["parch_imputer"] = parch_imputer
        else:
            parch_imputer = self.state_dependent_transforms["parch_imputer"]
            output = parch_imputer.transform(output)

        if "embarked_imputer" not in self.state_dependent_transforms.keys():
            embarked_imputer = NAImputer("embarked", method="mode")
            embarked_imputer.fit(output)
            output = embarked_imputer.transform(output)
            self.state_dependent_transforms["embarked_imputer"] = embarked_imputer
        else:
            embarked_imputer = self.state_dependent_transforms["embarked_imputer"]
            output = embarked_imputer.transform(output)

        if "room_imputer" not in self.state_dependent_transforms.keys():
            room_imputer = NAImputer("room", method="median")
            room_imputer.fit(output)
            output = room_imputer.transform(output)
            self.state_dependent_transforms["room_imputer"] = room_imputer
        else:
            room_imputer = self.state_dependent_transforms["room_imputer"]
            output = room_imputer.transform(output)

        if "deck_imputer" not in self.state_dependent_transforms.keys():
            deck_imputer = NAImputer("deck", method="mode")
            deck_imputer.fit(output)
            output = deck_imputer.transform(output)
            self.state_dependent_transforms["deck_imputer"] = deck_imputer
        else:
            deck_imputer = self.state_dependent_transforms["deck_imputer"]
            output = deck_imputer.transform(output)

        # Ensure string
        output["embarked"] = output["embarked"].astype(str)
        output["deck"] = output["deck"].astype(str)

        # One-hot Encode
        if "one_hot_encode_title" not in self.state_dependent_transforms.keys():
            title_lb = OneHotEncoder("title")
            title_lb.fit(output)
            output = title_lb.transform(output)
            self.state_dependent_transforms["one_hot_encode_title"] = title_lb
        else:
            title_lb = self.state_dependent_transforms["one_hot_encode_title"]
            output = title_lb.transform(output)

        if "one_hot_encode_embarked" not in self.state_dependent_transforms.keys():
            embark_lb = OneHotEncoder("embarked")
            embark_lb.fit(output)
            output = embark_lb.transform(output)
            self.state_dependent_transforms["one_hot_encode_embarked"] = embark_lb
        else:
            embark_lb = self.state_dependent_transforms["one_hot_encode_embarked"]
            output = embark_lb.transform(output)

        if "one_hot_encode_deck" not in self.state_dependent_transforms.keys():
            deck_lb = OneHotEncoder("deck")
            deck_lb.fit(output)
            output = deck_lb.transform(output)
            self.state_dependent_transforms["one_hot_encode_deck"] = deck_lb
        else:
            deck_lb = self.state_dependent_transforms["one_hot_encode_deck"]
            output = deck_lb.transform(output)

        # Lone passenger
        output["lone"] = 0
        output.loc[(output["sibsp"] + output["parch"] < 1), "lone"] = 1

        output.drop(columns=["name", "passengerid", "sex", "ticket", "cabin"], inplace=True)

        return output.reset_index(drop=True)

    @staticmethod
    def get_title(x):
        title = x.split(",")[1].split(".")[0].strip()

        if title.lower() not in ['capt', 'dr', 'major', 'master', 'miss', 'mr', 'mrs', 'ms', 'sir']:
            title = "other"

        return title.lower()
