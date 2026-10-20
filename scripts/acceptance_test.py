"""Real PostgreSQL + Azure RAG + Ollama acceptance tests in an isolated disposable database.
Only child processes created by this script are killed. Normal app history is not modified.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]


class Lab:
    def __init__(self, env, logs):
        self.env, self.logs, self.processes, self.handles = env, logs, {}, []

    def start(self, name, port, module, overrides=None):
        self.stop(name)
        environment = {**os.environ, **self.env, **(overrides or {})}
        handle = (self.logs / f'{name}-{time.time_ns()}.log').open('w')
        self.handles.append(handle)
        self.processes[name] = subprocess.Popen([
            sys.executable, '-m', 'uvicorn', module, '--host', '127.0.0.1', '--port', str(port),
            '--no-access-log', '--timeout-graceful-shutdown', '1',
        ], cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)

    def stop(self, name, hard=False):
        process = self.processes.pop(name, None)
        if process and process.poll() is None:
            process.kill() if hard else process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def close(self):
        for name in list(self.processes):
            self.stop(name)
        for handle in self.handles:
            handle.close()


def wait_for(predicate, timeout=120, description='condition'):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except (httpx.HTTPError, KeyError, ValueError):
            pass
        time.sleep(.2)
    raise TimeoutError(f'Timed out waiting for {description}')


def session(client):
    response = client.post('/api/sessions')
    response.raise_for_status()
    return response.json()['session_id']


def submit(client, sid, question):
    payload = {'request_id': str(uuid4()), 'question': question, 'language': 'en'}
    for _ in range(2):
        response = client.post(f'/api/sessions/{sid}/questions', json=payload)
        response.raise_for_status()
    return payload['request_id']


def completed(client, request):
    result = client.get(f'/api/requests/{request}').json()
    if result.get('status') == 'failed':
        raise AssertionError(f"Unexpected request failure: {result.get('error')}")
    return result if result.get('status') == 'completed' else None


def first_partial(client, sid, request):
    with client.stream('GET', f'/api/sessions/{sid}/events', timeout=120) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith('data: '):
                continue
            data = json.loads(line[6:])
            jobs = data if isinstance(data, list) else [data]
            for job in jobs:
                if job['request_id'] == request and job.get('partial_answer'):
                    return job
                if job['request_id'] == request and job['status'] in {'completed', 'failed'}:
                    raise AssertionError('No partial frame was observed before the terminal state')
    raise AssertionError('SSE closed before a partial answer')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='.env.mac')
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    env = {key: value for key, value in dotenv_values(ROOT / args.env).items() if value is not None}
    url = make_url(env['DATABASE_URL'])
    if url.get_backend_name() != 'postgresql':
        raise SystemExit('This acceptance test requires PostgreSQL, not SQLite.')
    for port in [18000, 18002, 18003]:
        with socket.socket() as connection:
            connection.bind(('127.0.0.1', port))
    name = 'rag_acceptance_' + uuid4().hex[:12]
    directory = ROOT / 'data' / 'reports' / f'acceptance-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}'
    directory.mkdir(parents=True)
    admin = create_engine(url, isolation_level='AUTOCOMMIT')
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    test_url = url.set(database=name).render_as_string(hide_password=False)
    env.update({'DATABASE_URL': test_url, 'INFERENCE_URL': 'http://127.0.0.1:18002',
                'LEASE_SECONDS': '6', 'WORKER_POLL_SECONDS': '.1', 'RETRY_BASE_SECONDS': '.1',
                'JOB_TIMEOUT_SECONDS': '120', 'WORKER_CONCURRENCY': '2', 'CACHE_TTL_SECONDS': '3600'})
    lab = Lab(env, directory)
    checks = []
    def passed(label):
        checks.append(label)
        print('PASS:', label, flush=True)
    nonce = uuid4().hex[:8]
    try:
        lab.start('app', 18000, 'services.app.main:app')
        with httpx.Client(base_url='http://127.0.0.1:18000', timeout=20) as client:
            wait_for(lambda: client.get('/api/health').json().get('application'), description='application startup')
            sid = session(client)
            question = f'What is double quantization in QLoRA? Answer in three sentences. Test {nonce}.'
            request = submit(client, sid, question)
            time.sleep(1)
            queued = client.get(f'/api/requests/{request}').json()
            assert queued['status'] == 'queued' and queued['attempts'] == 0
            passed('Offline inference preserves the durable queue without consuming retries')
            lab.stop('app')
            lab.start('app', 18000, 'services.app.main:app')
            wait_for(lambda: client.get(f'/api/requests/{request}').status_code == 200, description='app restart')
            assert client.get(f'/api/sessions/{sid}/messages').json()[0]['request_id'] == request
            passed('Application restart preserves accepted requests and conversation history')
            lab.start('inference', 18002, 'services.inference.main:app')
            partial = first_partial(client, sid, request)
            assert partial['status'] == 'generating'
            answer = wait_for(lambda: completed(client, request), args.timeout, 'first real answer')
            assert answer['answer'] and answer['sources']
            passed('Actual model tokens reach SSE before completion; response has sources')
            again_sid = session(client)
            again = submit(client, again_sid, question)
            cached = wait_for(lambda: completed(client, again), args.timeout, 'cache answer')
            assert cached['cache_hit'] and cached['answer'] == answer['answer']
            passed('A repeated question with identical context hits the persistent cache')
            sid2 = session(client)
            request2 = submit(client, sid2, f'Explain paged optimizers in QLoRA in five sentences. Test {nonce}.')
            first_partial(client, sid2, request2)
            lab.stop('inference', hard=True)
            wait_for(lambda: client.get(f'/api/requests/{request2}').json().get('status') == 'queued',
                     args.timeout, 'retry after inference crash')
            interrupted = client.get(f'/api/requests/{request2}').json()
            assert not interrupted['answer'] and not interrupted['partial_answer']
            lab.start('inference', 18002, 'services.inference.main:app')
            recovered = wait_for(lambda: completed(client, request2), args.timeout, 'inference recovery')
            assert recovered['attempts'] >= 2 and len(client.get(f'/api/sessions/{sid2}/messages').json()) == 1
            passed('Inference hard crash during streaming clears partial output and recovers once')
            sid3 = session(client)
            request3 = submit(client, sid3, f'Explain multi-head attention in five sentences. Test {nonce}.')
            first_partial(client, sid3, request3)
            lab.stop('app', hard=True)
            lab.start('app', 18000, 'services.app.main:app')
            recovered = wait_for(lambda: completed(client, request3), args.timeout, 'expired lease recovery')
            assert recovered['attempts'] >= 2 and len(client.get(f'/api/sessions/{sid3}/messages').json()) == 1
            passed('Hard application crash recovers an expired lease without duplicate history')
            lab.start('slow-peer', 18003, 'tests.fault_peer:app')
            lab.stop('app')
            lab.start('app', 18000, 'services.app.main:app', {
                'INFERENCE_URL': 'http://127.0.0.1:18003', 'JOB_TIMEOUT_SECONDS': '1', 'MAX_ATTEMPTS': '2'})
            wait_for(lambda: client.get('/api/health').json().get('inference'), description='deadline test startup')
            sid4 = session(client)
            request4 = submit(client, sid4, f'Explain retrieval augmented generation in detail. Test deadline {nonce}.')
            failed = wait_for(lambda: (result if (result := client.get(f'/api/requests/{request4}').json())['status'] == 'failed' else None),
                              args.timeout, 'bounded timeout retries')
            assert failed['attempts'] == 2 and not failed['answer'] and not failed['partial_answer']
            passed('Deterministic deadline failures stop at the configured retry limit')
            lab.stop('app')
            lab.start('app', 18000, 'services.app.main:app')
            wait_for(lambda: client.get('/api/health').json().get('inference'), description='normal app startup')
            response = client.post(f'/api/requests/{request4}/retry')
            response.raise_for_status()
            wait_for(lambda: completed(client, request4), args.timeout, 'manual retry recovery')
            assert len(client.get(f'/api/sessions/{sid4}/messages').json()) == 1
            passed('Manual retry retains request_id and does not duplicate messages')
            other = client.get(f'/api/sessions/{sid2}/messages').json()
            assert all(job['session_id'] == sid2 and job['request_id'] == request2 for job in other)
            passed('Concurrent sessions retain separate histories')
        report = {'passed': True, 'checks': checks, 'database': 'isolated PostgreSQL',
                  'rag': 'configured real RAG node', 'inference': 'local Ollama via controlled API child processes',
                  'deadline_case': 'loopback delayed peer', 'model': env.get('LLM_MODEL')}
        (directory / 'result.json').write_text(json.dumps(report, indent=2))
        print(f'Report: {directory / "result.json"}', flush=True)
    finally:
        lab.close()
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


if __name__ == '__main__':
    main()
