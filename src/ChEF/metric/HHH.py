from tqdm import tqdm
from .utils import Base_Metric

class HHH_Metric(Base_Metric):

    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)

    def metric_func(self, answers):
        return dict(), answers