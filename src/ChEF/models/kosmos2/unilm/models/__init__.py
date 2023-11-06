import importlib
import os
from fairseq.models import import_models

models_dir = os.path.dirname(__file__)
import_models(models_dir, "models.kosmos2.unilm.models")