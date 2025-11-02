from datetime import datetime

def get_timestamp(format_str="%Y%m%d_%H%M%S"):
    return datetime.now().strftime(format_str)