import logging
import os
import colorlog
import re
import datetime
import pytz
import os

from colorama import init
from torch.utils.tensorboard import SummaryWriter

def combo_dict(d1, d2):
    d = {}
    for k in d1:
        d[k] = d1[k]
    for k in d2:
        d[k] = d2[k]
    return d

def ensure_dir(path):
    if not os.path.exists(path):
        print('making dir: {}'.format(path))
        os.makedirs(path)
    

def get_local_time():
    tz = pytz.timezone('Asia/Shanghai')
    cur = datetime.datetime.now(tz)
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur

def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += '%s : %.4f' % (str(metric), value) + '    '
    return result_str

log_colors_config = {
    'DEBUG': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}

class RemoveColorFilter(logging.Filter):
    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True

def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'

def init_logger(config, icc=False):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.
    Args:
        config (Config): An instance object of Config, used to record parameter information.
    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    init(autoreset=True)
    LOGROOT = './log/{}'.format(config['train_phase'])
    # dir_name = os.path.dirname(LOGROOT)
    ensure_dir(LOGROOT)
    
    if icc:
        logfilename = 'icc-{}-{}-{}.log'.format(config['model'], config['dataset'], get_local_time())
    else:
        logfilename = '{}-{}-{}.log'.format(config['model'], config['dataset'], get_local_time())

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    
    level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[sh, fh])

def get_tensorboard(logger, train_stage = 'ind'):
    r""" Creates a SummaryWriter of Tensorboard that can log PyTorch models and metrics into a directory for 
    visualization within the TensorBoard UI.
    For the convenience of the user, the naming rule of the SummaryWriter's log_dir is the same as the logger.

    Args:
        logger: its output filename is used to name the SummaryWriter's log_dir.
                If the filename is not available, we will name the log_dir according to the current time.

    Returns:
        SummaryWriter: it will write out events and summaries to the event file.
    """
    base_path = './log_tensorboard/{}'.format(train_stage)

    dir_name = None
    for handler in logger.handlers:
        if hasattr(handler, "baseFilename"):
            dir_name = os.path.basename(getattr(handler, 'baseFilename')).split('.')[0]
            break
    if dir_name is None:
        dir_name = '{}-{}'.format('model', get_local_time())

    dir_path = os.path.join(base_path, dir_name)
    writer = SummaryWriter(dir_path)
    return writer