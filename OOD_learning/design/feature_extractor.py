from .woe_encoder import WoeEncoder


class FeatureExtractor:

    """
    Code is based on the package

    Title: XuniVerse
    Author: Sundar Krishnan
    Date: 2020
    Availability: https://github.com/Sundar0989/XuniVerse
    """

    def __init__(self, selection_techniques):

        self.selection_techniques = selection_techniques

    def woe_information_value(self, X, y):

        clf = WoeEncoder()
        clf.fit(X, y)

        return clf.transform(X), clf.woe_bins, clf.iv_df

    def fit(self, X, y):

        self.use_features = X.columns
        self.no_of_features = int(len(self.use_features) / 2)
        self.feature_importances_, self.feature_votes_ = self.train(X, y)

        return self

    def train(self, X, y):

        dfs = []
        output_columns = []

        # run woe function
        if 'WOE' in self.selection_techniques:
            name = 'Information_Value'
            _, _, iv_df = self.woe_information_value(X, y)
            dfs.append(iv_df)
            output_columns.append(name)

