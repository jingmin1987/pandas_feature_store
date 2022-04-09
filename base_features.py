import abc
import pandas as pd

class BaseFeature(abc.ABC):
    @property
    @abc.abstractmethod
    def column_name(self):
        pass

    @property
    @abc.abstractmethod
    def statement(self):
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(**self.statement)


class SimpleFeature(BaseFeature):
    def __init__(self, name):
        self.name = name

    @property
    def column_name(self):
        return self.name

    @property
    def statement(self):
        return {}
