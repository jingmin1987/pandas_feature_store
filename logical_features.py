from base_features import BaseFeature


class GtEqFeature(BaseFeature):
    """
    Greater than or equal feature

    :param first_feature: Left feature
    :param second_feature: Right feature
    """

    def __init__(self, first_feature, second_feature):
        self.first_feature = first_feature
        self.second_feature = second_feature

    @property
    def column_name(self):
        return f'{self.first_feature.column_name}_gt_or_eq_{self.second_feature.column_name}'

    @property
    def statement(self):
        return {self.column_name:
                    lambda df: (df[self.first_feature.column_name] >= df[self.second_feature.column_name]) * 1}
