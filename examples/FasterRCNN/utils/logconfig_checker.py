import os
import re


class LogConfigChecker:
    def __init__(self, load_weight, default_log_subdir=None, log_root="/root/dentalpoc/logs"):
        self.default_log_subdir = default_log_subdir
        self.log_root = log_root
        self.load_weight = load_weight

    def ask_for_log_subdir(self):
        log_subdir = self.default_log_subdir
        log_root = self.log_root

        if log_subdir is None:
            log_subdir = input("Please specify log_subdir, "
                               "the record will be kept under {}".format(log_root) +
                               "/{log_subdir}_{TIMESTAMP}\n")
        return log_subdir

    def verify_logdir(self, log_subdir):
        log_root = self.log_root
        logdir = os.path.join(log_root, log_subdir)
        branch_code = 1

        print("checking >>> {}".format(logdir))

        load_weight = None
        if os.path.exists(logdir):
            is_pickup = input("log subdir already exists, do you want to continue the training? (yes/no) \n")
            if re.match("^yes$", is_pickup.lower()):
                load_weight = os.path.join(logdir, "checkpoint")
            else:
                print("Please provide a different log_dir to start a new training job")
                branch_code = -1
        else:
            input("will create new log_dir: {}".format(logdir))

        return logdir, load_weight, branch_code

    def verify_load_weight(self, load_weight):
        if os.path.exists(load_weight):
            check = input("training will start from:"
                          "\n{}\nConfirm? (yes/no)\n".format(load_weight))
        else:
            check = input("Could not find weight to load"
                          "and the training will start from "
                          "scratch \nConfirm? (yes/no)\n".format(load_weight))
        return check

    def execute(self):
        load_weight = None
        logdir = 'test'
        branch_code = -1
        while branch_code < 0:
            log_subdir = self.ask_for_log_subdir()
            logdir, load_weight, branch_code = self.verify_logdir(log_subdir)

        load_weight = load_weight if load_weight is not None else self.load_weight
        self.verify_load_weight(load_weight)

        return logdir, load_weight
