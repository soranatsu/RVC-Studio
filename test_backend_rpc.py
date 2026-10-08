from pathlib import Path
import json, math, struct, tempfile, wave, time
import time as _time
import studio_backend
from studio_backend import StudioBackend

ROOT = Path(__file__).resolve().parent

def slow_worker(conn, root):
    # Controlled hang: it deliberately never reads the RPC pipe.
    _time.sleep(15)

def healthy_worker(conn, root):
    while True:
        if not conn.poll(.05): continue
        message = conn.recv(); command = message.get('cmd')
        if command == 'shutdown': return
        if command == 'stop_live': conn.send({'kind':'live','state':'stopped'}); continue
        if command == 'start_live': conn.send({'kind':'live','state':'started','result':{}}); continue

def wav(path):
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
        f.writeframes(b''.join(struct.pack('<h', int(12000*math.sin(2*math.pi*440*i/16000))) for i in range(16000)))

if __name__ == '__main__':
    work = Path(tempfile.mkdtemp(prefix='rpc-test-', dir=ROOT)); src = work/'tone.wav'; wav(src)
    events=[]; backend=StudioBackend(ROOT, events.append)
    backend.probe_devices(); deadline=time.time()+15
    while time.time()<deadline and not any(e.job_id=='devices' for e in events): time.sleep(.05)
    assert any(e.job_id=='devices' and e.state=='ready' for e in events), events
    model=ROOT/'assets/weights/aiyi.pth'; index=ROOT/'assets/indices/aiyi.index'
    backend.validate_voice(model,index); deadline=time.time()+30
    while time.time()<deadline and not any(e.job_id=='voice' for e in events): time.sleep(.05)
    assert any(e.job_id=='voice' and e.state=='validated' for e in events), events
    jobdir=work/'job'; jobdir.mkdir(); job=jobdir/'job.json'
    job.write_text(json.dumps({'operation':'spectrum','source':str(src),'output_dir':str(work/'out')}), encoding='utf8')
    jid,f=backend.run_job(job); f.result(timeout=30); assert (work/'job'/'status.json').is_file()
    original = studio_backend._worker_main
    studio_backend._worker_main = healthy_worker
    healthy = StudioBackend(ROOT, events.append); healthy_pid = healthy._process.pid
    healthy.start_live({})
    time.sleep(.1); started = time.monotonic(); healthy.stop_live()
    while time.monotonic()-started < 2 and not any(e.state == 'stopped' for e in events): time.sleep(.01)
    assert any(e.state == 'stopped' for e in events)
    time.sleep(6); assert healthy._process.pid == healthy_pid
    t0=time.monotonic()
    for _ in range(1000): healthy.update_live({'pitch': 6})
    assert time.monotonic()-t0 < .2
    healthy.shutdown(); studio_backend._worker_main = original
    studio_backend._worker_main = slow_worker
    hung = StudioBackend(ROOT)
    old_pid = hung._process.pid
    hang_id, hang_future = hung.run_job(job)
    studio_backend._worker_main = original
    started = time.monotonic(); assert hung.cancel(hang_id)
    while time.monotonic() - started < 6.5 and not hang_future.done(): time.sleep(.05)
    assert hang_future.done() and hang_future.result() is None
    assert hung._process.pid != old_pid
    jid2, future2 = hung.run_job(job); future2.result(timeout=30)
    hung.shutdown(); backend.shutdown(); print('BACKEND_RPC_PASS')
