'''
Created on Jan 12, 2023

@author: rkaria
'''

import config
import os
import shutil
from utils.file_utils import FileUtils
import csv
import threading
import time

class ResultLogger:
    
    M = "Metadata"
    ITR = "Interaction"
    SR = "Success Rate"
    VD_SAMPLES = "Variational Difference (Samples)"
    VD_GT = "Best VD"
    ET = "Elapsed Time"
    LAO_C = "LAO (Costs)"
    LAO_R = "LAO (Success Rate)"
    LAO_AGG_C = "LAO (Aggregate Costs)"
    LAO_AGG_R = "LAO (Aggregate Success Rate)"
    
    CSV_SEPARATOR = ";"
    CSV_HEADER = [M, ITR, SR, VD_SAMPLES, VD_GT, ET, LAO_C, LAO_R,
                  LAO_AGG_C, LAO_AGG_R]
    
    @staticmethod
    def get_name(prefix, *args):
        
        string = prefix
        
        for arg in args:
            
            string += "_%s" % (arg)

        return string.lower()
    
    @staticmethod
    def create_csv_files(results_dir, filename="results.csv",
                         clean_dir=False, clean_file=True):
        

        FileUtils.initialize_directory(results_dir, clean=clean_dir)
        
        if not filename.endswith(".csv"):
            results_csvfilepath = "%s/%s.csv" % (results_dir, filename)
        else:
            results_csvfilepath = "%s/%s" % (results_dir, filename)
        
        assert not os.path.exists(results_csvfilepath) \
            or not os.path.isdir(results_csvfilepath)
            

        write_header = True
        results_filehandle = open(results_csvfilepath, "w")
        results_csvwriter = csv.DictWriter(results_filehandle,
                                       delimiter=ResultLogger.CSV_SEPARATOR,
                                       fieldnames=ResultLogger.CSV_HEADER)
        
        if write_header:
            results_csvwriter.writeheader()
            results_filehandle.flush()
            
        return results_filehandle, results_csvwriter
    
    def __init__(self, results_dir, filename="results.csv",
                        clean_dir=False, clean_file=True):
        
        self.results_filehandle, self.results_csvwriter = \
            ResultLogger.create_csv_files(results_dir, 
                                          filename, 
                                          clean_dir, clean_file)
        
    def log_results(self, metadata, itr, success_rate, vd_samples,
                    vd_gt, elapsed_time, total_cost=None,  
                    total_reward=None,  agg_cost=None, 
                    agg_reward=None):
        
        self.results_csvwriter.writerow({
                ResultLogger.M: metadata,
                ResultLogger.ITR: itr,
                ResultLogger.SR: success_rate,
                ResultLogger.VD_SAMPLES: vd_samples,
                ResultLogger.VD_GT: vd_gt,
                ResultLogger.ET: elapsed_time,
                ResultLogger.LAO_C: total_cost,
                ResultLogger.LAO_R: total_reward,
                ResultLogger.LAO_AGG_C: agg_cost,
                ResultLogger.LAO_AGG_R: agg_reward,
            })
        
        self.results_filehandle.flush()


class DiffResultsLogger(threading.Timer):

    CSV_SEPARATOR = ";"

    METADATA = "Metadata"
    HORIZON = "Horizon"
    TASK_NAME = "Task Name"
    TASK_NO = "Task Number"
    NUM_SIMULATIONS = "# of Simulations"
    SIM_BUDGET = "Simulator Budget"
    SUCCESS_RATE = "Avg. Success Rate"
    REWARD = "Avg. Reward"
    TIME = "Elapsed Time (s)"
    STEPS = "Total Steps"
    GOALS = "Total Goals Reached"

    STEPS_HEADER = [
        METADATA,
        HORIZON,
        NUM_SIMULATIONS,
        SIM_BUDGET,
        TASK_NAME,
        TASK_NO,
        STEPS,
        GOALS,
        SUCCESS_RATE,
        REWARD
    ]

    TIME_HEADER = [
        METADATA,
        HORIZON,
        NUM_SIMULATIONS,
        SIM_BUDGET,
        TASK_NAME,
        TASK_NO,
        TIME,
        GOALS,
        SUCCESS_RATE,
        REWARD
    ]

    @staticmethod
    def get_csv_names(results_dir, name):

        steps_file = "%s/%s_steps_vs_tasks.csv" % (results_dir, name)
        time_file = "%s/%s_time_vs_tasks.csv" % (results_dir, name)

        return steps_file, time_file

    def __init__(self, results_dir, name,
                 logging_data_func,
                clean=False,
                step_log_interval=100,
                logging_time_in_sec=1):

        self.step_log_interval = step_log_interval
        self.logging_time_in_sec = logging_time_in_sec
        self.time_iteration = 1
        self.total_steps = 0

        FileUtils.initialize_directory(results_dir, clean=False)

        mode = "w" if clean else "a"
        steps_file, time_file = DiffResultsLogger.get_csv_names(results_dir,
                                                                name)
        self.steps_fh = open(steps_file, mode)
        self.time_fh = open(time_file, mode)

        self.steps_csv = csv.DictWriter(self.steps_fh,
            delimiter=DiffResultsLogger.CSV_SEPARATOR,
            fieldnames=DiffResultsLogger.STEPS_HEADER)

        self.time_csv = csv.DictWriter(self.time_fh,
            delimiter=DiffResultsLogger.CSV_SEPARATOR,
            fieldnames=DiffResultsLogger.TIME_HEADER)

        if clean:

            self.steps_csv.writeheader()
            self.time_csv.writeheader()
            self.flush()

        self.done = False

        super(DiffResultsLogger, self).__init__(self.logging_time_in_sec,
                                                self.log_time_data)

        self.logging_data_func = logging_data_func

    def flush(self):

        self.steps_fh.flush()
        self.time_fh.flush()

    def log_step_data(self, total_steps):

        self.total_steps += 1
        if self.total_steps % self.step_log_interval == 0:

            data = self.logging_data_func()
            data[DiffResultsLogger.STEPS] = self.total_steps

            self.steps_csv.writerow(data)
            self.steps_fh.flush()

    def log_time_data(self):

        data = self.logging_data_func()

        logging_time = self.time_iteration * self.logging_time_in_sec
        self.time_iteration += 1

        data[DiffResultsLogger.TIME] = logging_time

        self.time_csv.writerow(data)
        self.time_fh.flush()

    def run(self):

        # https://stackoverflow.com/questions/12435211/threading-timer-repeat-function-every-n-seconds
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)
