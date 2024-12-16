import os
import csv
from pathlib import Path


class Logger:
    def __init__(self, dir, fname):
        # Create dictionary if does not exist
        Path(dir).mkdir(parents=True, exist_ok=True)
        self.fname = dir + "/" + fname

        if os.path.exists(self.fname):
            os.remove(self.fname)

        self.res = {}
        self.fieldnames = None

    def log_custom(self, name, value):
        self.res[name] = value

    def add_record_to_log_file(self):
        # if the file exists - append to the end
        if os.path.isfile(self.fname):
            with open(self.fname, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(self.res.keys()))
                writer.writerow(self.res)
        else:

            with open(self.fname, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(self.res.keys()))
                writer.writeheader()
                writer.writerow(self.res)
        self.res = {}
        return

