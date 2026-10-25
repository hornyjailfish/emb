import logging
import sys
from importlib import util

logger = logging.getLogger(__name__)

# Core always required. Backend-specific libs are checked in check_deps()
# against Settings.model_backend so Jina-only hosts need not install MegaLoc
# extras and vice versa.
REQUIRED_LIBS = ["torch", "fastapi", "surrealdb", "pydantic", "numpy"]
JINA_LIBS = ["sentence_transformers"]
MEGALOC_LIBS = ["torchvision", "huggingface_hub", "safetensors"]
OPTIONAL_LIBS = ["win32api", "win32con", "win32process"]
UTILITY_LIBS = ["colorlog"]

missing_libs = [lib for lib in REQUIRED_LIBS if util.find_spec(lib) is None]
missing_optional_libs = [lib for lib in OPTIONAL_LIBS if util.find_spec(lib) is None]
missing_utility_libs = [lib for lib in UTILITY_LIBS if util.find_spec(lib) is None]

def quiet_third_party_loggers():
    """Silence INFO/DEBUG chatter from HTTP and model-download libraries."""
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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
    quiet_third_party_loggers()

def setup_process_prio():
    if set(missing_optional_libs) & {"win32api", "win32con", "win32process"}:
        return
    else:
        import win32api
        import win32con
        import win32process
        priority = win32process.BELOW_NORMAL_PRIORITY_CLASS
        label = {
            win32process.BELOW_NORMAL_PRIORITY_CLASS: "BELLOW NORMAL",
            win32process.ABOVE_NORMAL_PRIORITY_CLASS: "ABOVE NORMAL",
            win32process.HIGH_PRIORITY_CLASS: "HIGH",
            win32process.IDLE_PRIORITY_CLASS: "IDLE",
            win32process.NORMAL_PRIORITY_CLASS: "NORMAL",
            win32process.REALTIME_PRIORITY_CLASS: "REALTIME",
        }
        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, priority)
        logger.info(f"Системный приоритет процесса успeшно установлен {label[priority]}.")

def check_deps():
    if missing_libs:
        logger.critical("Отсутствуют обязательные зависимости!")
        for lib in missing_libs:
            logger.error(f"  - {lib} не установлена\n")
        logger.info("\nУстановите их командой: pip install " + " ".join(missing_libs))
        sys.exit(1)

    # Backend-specific imports — resolve settings lazily (needs .env).
    backend = "megaloc"
    try:
        from src.config import get_settings

        backend = (get_settings().model_backend or "megaloc").strip().lower()
    except Exception as e:
        logger.warning("Could not read model_backend for dep check: %s", e)

    extra = MEGALOC_LIBS if backend in ("megaloc", "vpr") else JINA_LIBS
    missing_backend = [lib for lib in extra if util.find_spec(lib) is None]
    if missing_backend:
        logger.critical(
            "Отсутствуют зависимости для model_backend=%s!", backend
        )
        for lib in missing_backend:
            logger.error(f"  - {lib} не установлена\n")
        logger.info(
            "\nУстановите их командой: pip install " + " ".join(missing_backend)
        )
        sys.exit(1)

    if missing_optional_libs:
        logger.warning("Отсутствуют необязательные зависимости!\n")
        logger.info("Установите их командой: pip install " + " ".join(missing_optional_libs))
    if missing_utility_libs:
        logger.info("Отсутствуют дополнительные зависимости!\n")
        logger.info("Установите их командой: pip install " + " ".join(missing_utility_libs))
