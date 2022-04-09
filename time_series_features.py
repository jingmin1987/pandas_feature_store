"""
Time series features used widely in my trading related projects
"""
import numpy as np
from base_features import BaseFeature


class LagFeature(BaseFeature):
    def __init__(self, base_feature, lag):
        self.base_feature = base_feature
        self.lag = lag

    @property
    def column_name(self):
        return f'{self.base_feature.column_name}_{self.lag}b_lag'

    @property
    def statement(self):
        return {self.column_name: lambda df: df[self.base_feature.column_name].shift(self.lag)}


class ReturnFeature(BaseFeature):
    def __init__(self, base_feature, duration=1):
        self.base_feature = base_feature
        self.duration = duration

    @property
    def column_name(self):
        return f'{self.base_feature.column_name}_{self.duration}b_return'

    @property
    def statement(self):
        return {self.column_name:
                    lambda df: (df[self.base_feature.column_name] / df[self.base_feature.column_name].shift(
                        self.duration)) ** np.sign(self.duration) - 1}


class WeightedAverageFeature(BaseFeature):
    """
    Weighted average of a metric

    :param first_feature: Metric to be averaged
    :param second_feature: Weight to use
    :param duration: Lookback window length
    :param win_type: Window type for the lookback window
    """

    def __init__(self, first_feature, second_feature, duration, win_type=None):
        self.first_feature = first_feature
        self.second_feature = second_feature
        self.duration = duration
        self.win_type = win_type

    @property
    def column_name(self):
        win_type = (self.win_type and f'_{self.win_type}') or ''

        return f'{self.first_feature.column_name}{win_type}_weighted_by_{self.second_feature.column_name}_over_{self.duration}b'

    @property
    def statement(self):
        # Shorter name to make code more concise
        metric = self.first_feature.column_name
        weight = self.second_feature.column_name
        duration = self.duration
        win_type = self.win_type

        return {self.column_name:
                    lambda df: (df[metric] * df[weight]).rolling(duration, win_type=win_type).sum() / df[
                        weight].rolling(duration).sum()}


class SimpleAverageFeature(BaseFeature):
    """
    Simple average of a metric

    :param base_feature: Feature to be averaged
    :param duration: Lookback window length
    :param win_type: Window type for the lookback window
    """

    def __init__(self, base_feature, duration, win_type=None):
        self.base_feature = base_feature
        self.duration = duration
        self.win_type = win_type

    @property
    def column_name(self):
        win_type = (self.win_type and f'_{self.win_type}') or ''

        return f'{self.base_feature.column_name}{win_type}_avg_over_{self.duration}b'

    @property
    def statement(self):
        metric = self.base_feature.column_name
        duration = self.duration
        win_type = self.win_type

        return {self.column_name:
                    lambda df: df[metric].rolling(duration, win_type=win_type).mean()}


class VolatilityFeature(BaseFeature):
    """
    Calculates the population (not sample) standard deviation of a metric for a given period

    :param base_feature: Feature to be averaged
    :param duration: Lookback window length, absolute min is 2
    :param win_type: Window type for the lookback window
    """

    def __init__(self, base_feature, duration, win_type=None):
        self.base_feature = base_feature
        self.duration = max(abs(duration), 2) * np.sign(duration)
        self.win_type = win_type

    @property
    def column_name(self):
        win_type = (self.win_type and f'_{self.win_type}') or ''

        return f'{self.base_feature.column_name}{win_type}_volatility_over_{self.duration}b'

    @property
    def statement(self):
        metric = self.base_feature.column_name
        duration = self.duration
        win_type = self.win_type

        return {self.column_name:
                    lambda df: df[metric].rolling(duration, win_type=win_type).std(ddof=0)}
