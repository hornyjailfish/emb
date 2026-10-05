import logging
import sys
from importlib import util

logger = logging.getLogger(__name__)

REQUIRED_LIBS = ["torch", "fastapi", "surrealdb", "pydantic", "sentence_transformers", "numpy"]
OPTIONAL_LIBS = ["win32api", "win32con", "win32process"]
UTILITY_LIBS = ["colorlog"]

missing_libs = [lib for lib in REQUIRED_LIBS if util.find_spec(lib) is None]
missing_optional_libs = [lib for lib in OPTIONAL_LIBS if util.find_spec(lib) is None]
missing_utility_libs = [lib for lib in UTILITY_LIBS if util.find_spec(lib) is None]

def setup_logging(log_level):
    root_logger = logging.getLogger()
    if "colorlog" in missing_utility_libs:
        root_logger.info("Для цветного вывода логов требуется библиотека colorlog")
    else:
        import colorlog
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            fmt="%(log_color)s%(levelname)s%(reset)s:%(name)s\t %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
        root_logger.addHandler(handler)

    root_logger.setLevel(log_level)

def setup_process_prio():
    if set(missing_optional_libs) & {"win32api", "win32con", "win32process"}:
        return
    else:
        import win32api
        import win32con
        import win32process
        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.IDLE_PRIORITY_CLASS)
        logger.info("Системный приоритет процесса успешно снижен до IDLE.")

def check_deps():
    if missing_libs:
        logger.critical("Отсутствуют обязательные зависимости!")
        for lib in missing_libs:
            logger.error(f"  - {lib} не установлена\n")
        logger.info("\nУстановите их командой: pip install " + " ".join(missing_libs))
        sys.exit(1)
    if missing_optional_libs:
        logger.warning("Отсутствуют необязательные зависимости!\n")
        logger.info("Установите их командой: pip install " + " ".join(missing_optional_libs))
    if missing_utility_libs:
        logger.info("Отсутствуют дополнительные зависимости!\n")
        logger.info("Установите их командой: pip install " + " ".join(missing_utility_libs))
