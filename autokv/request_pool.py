"""有界 HTTP 并发；预算、重试与结果写入全部由调用线程处理。"""
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import time

from autokv.client import VllmHttpError


def evaluate_requests(samples, client, concurrency, on_send, on_error, on_response):
    queue = deque((sample, 0, None) for sample in samples)
    pending = {}
    failure = None
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while queue or pending:
            try:
                while queue and len(pending) < concurrency and failure is None:
                    sample, retry, started = queue.popleft()
                    started = started if started is not None else time.monotonic()
                    on_send(retry)
                    future = pool.submit(client.chat_complete, sample["user_prompt"], sample["max_tokens"])
                    pending[future] = (sample, retry, started)
            except BaseException as exc:
                failure = failure or exc
            if not pending:
                break
            try:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
            except BaseException as exc:
                # 中断后不再派发，仍回收已发出请求；HTTP 本身有超时。
                failure = failure or exc
                continue
            for future in done:
                sample, retry, started = pending.pop(future)
                try:
                    response = future.result()
                except (VllmHttpError, TimeoutError) as exc:
                    try:
                        on_error(sample, retry, exc)
                    except BaseException as write_error:
                        failure = failure or write_error
                    retryable = not retry and (not isinstance(exc, VllmHttpError) or exc.status is None or exc.status >= 500)
                    if retryable and failure is None:
                        queue.appendleft((sample, retry+1, started))
                    else:
                        failure = failure or exc
                except BaseException as exc:
                    failure = failure or exc
                else:
                    try:
                        on_response(sample, response, started, retry)
                    except BaseException as exc:
                        failure = failure or exc
    if failure is not None:
        raise failure
