import asyncio
import logging
from uuid import UUID

from surrealdb import errors

logger = logging.getLogger(__name__)

def should_handle(data: list[str], processing_batch: list) -> bool:
    return True


async def handle_diff(db, uuid):
    try:
        generator = await db.subscribe_live(uuid)
        async for data in generator:
            logger.info(f"Got new task: {data}")
            # if should_handle(data, config):
                # return data
            # else:
                # logger.info(f"Ignoring change: {data}")
                # return None
    except Exception as e:
        logger.error(f"Error in sub: {e}")
        raise

async def sub(db, uuid: str | UUID):
    logger.info(f"Subscribing to live changes for {uuid}")
    try:
        task = asyncio.create_task(handle_diff(db, uuid), name=uuid.__str__())
        # res = await asyncio.wait([task])
        return task
    except asyncio.CancelledError:
        logger.warning("Sub canceled")
    except Exception as e:
        logger.error(f"Error in sub: {e}")
        raise



async def close_live_queries(db, tasks: set[asyncio.Task]):
    """Отменяет все фоновые задачи, кроме текущей."""
    try:
        current_task = asyncio.current_task()
        for task in tasks:
            if task is not current_task:
                logger.info(f"Cancelling task {task.get_name()}")
                await db.kill(task.get_name())
                task.cancel()
        asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All background tasks cleaned up successfully.")
    except asyncio.exceptions.CancelledError:
        pass
    except errors.NotAllowedError:
        pass
