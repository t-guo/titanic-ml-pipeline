import numpy as np
import pandas as pd
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, StandardScaler

TITLE_DICTIONARY = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
}

LOGGER = logging.getLogger('luigi-interface')


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


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, method="mean", target_value=None, group_col=None):
        self.target_col = target_col
        self.target_value = target_value
        self.method = method
        self.group_values_df = None
        self.group_col = group_col

    def fit(self, data):
        if self.group_col:
            if self.group_values_df is None:
                if self.method == "mean":
                    self.group_values_df = data.groupby(self.group_col)[self.target_col].mean().reset_index()
                elif self.method == "median":
                    self.group_values_df = data.groupby(self.group_col)[self.target_col].median().reset_index()
                elif self.method == "mode":
                    self.group_values_df = data.groupby(self.group_col)[self.target_col].mode().reset_index()
                else:
                    raise NotImplementedError("Method not implemented, must be one of ['mean', 'mode', 'static']")
        else:
            if self.target_value is None:
                if self.method == "mean":
                    self.target_value = data[self.target_col].mean()
                elif self.method == "median":
                    self.target_value = data[self.target_col].median()
                elif self.method == "mode":
                    self.target_value = data[self.target_col].mode()[0]
                else:
                    raise NotImplementedError("Method not implemented, must be one of ['mean', 'mode', 'static']")

    def transform(self, data):
        output = data.copy()

        if self.group_col:
            all_data = pd.merge(output, self.group_values_df, how='left', on=self.group_col, suffixes=['', '_imputed'])
            idx = pd.isnull(output[self.target_col])
            output.loc[idx, self.target_col] = all_data.loc[idx, self.target_col + "_imputed"]
        else:
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
        output = self.state_dependent_transformation(output, "age_imputer", Imputer("age", method="mean", group_col=["pclass", "sex", "title"]))
        output = self.state_dependent_transformation(output, "sibsp_imputer", Imputer("sibsp", method="median"))
        output = self.state_dependent_transformation(output, "parch_imputer", Imputer("parch", method="median"))
        output = self.state_dependent_transformation(output, "embarked_imputer", Imputer("embarked", method="mode"))
        output = self.state_dependent_transformation(output, "room_imputer", Imputer("room", method="median"))
        output = self.state_dependent_transformation(output, "deck_imputer", Imputer("deck", method="static", target_value="N"))

        # Family size
        output["family_size"] = output["sibsp"] + output["parch"] + 1

        # introducing other features based on the family size
        output['single'] = output['family_size'].map(lambda s: 1 if s == 1 else 0)
        output['small_family'] = output['family_size'].map(lambda s: 1 if 2 <= s <= 4 else 0)
        output['large_family'] = output['family_size'].map(lambda s: 1 if 5 <= s else 0)

        # Adjust fare by family size
        output["fare"] = output["fare"]/output["family_size"]

        # Impute adjusted fare
        output = self.state_dependent_transformation(output, "fare_imputer", Imputer("fare", group_col=["pclass", "deck"]))

        # Ensure string
        output["embarked"] = output["embarked"].astype(str)
        output["deck"] = output["deck"].astype(str)

        # One-hot Encode
        output = self.state_dependent_transformation(output, "one_hot_encode_title", OneHotEncoder("title"))
        output = self.state_dependent_transformation(output, "one_hot_encode_deck", OneHotEncoder("deck"))
        output = self.state_dependent_transformation(output, "one_hot_encode_embarked", OneHotEncoder("embarked"))

        # Drop columns
        output.drop(columns=["name", "passengerid", "sex", "ticket", "cabin"], inplace=True)

        # Preserve column order
        if "column_order" not in self.state_dependent_transforms.keys():
            column_order = list(output.columns)
            self.state_dependent_transforms["column_order"] = column_order
        else:
            column_order = self.state_dependent_transforms["column_order"]

        LOGGER.info("Column order: {}".format(",".join(column_order)))
        output = output[column_order]

        # Scale features
        output = self.state_dependent_transformation(output, "standard_scaler", StandardScaler())

        return output

    def state_dependent_transformation(self, data, key, transform_obj=None):
        if key not in self.state_dependent_transforms.keys():
            transform_obj.fit(data)
            self.state_dependent_transforms[key] = transform_obj
            return transform_obj.transform(data)
        else:
            return self.state_dependent_transforms[key].transform(data)

    @staticmethod
    def get_title(x, title_dictionary=TITLE_DICTIONARY):
        title = x.split(",")[1].split(".")[0].strip()

        return title_dictionary[title].lower()
