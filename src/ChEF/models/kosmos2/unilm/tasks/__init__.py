import argparse
import importlib
import os
from fairseq.tasks import import_tasks

tasks_dir = os.path.dirname(__file__)
import_tasks(tasks_dir, "ChEF.models.kosmos2.unilm.tasks")
