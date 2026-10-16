"""Own the deployment's C++ bridge; only explicit acquire starts --control."""
import os
import errno
import socket
import subprocess
import sys
import time

from core.robot import DEPLOY
from hybrid_deploy import protocol as p


def build_bridge():
    env = dict(os.environ, HYBRID_DEPLOY_PYTHON=sys.executable)
    subprocess.run(['bash', str(DEPLOY / 'build.sh')], env=env, check=True)


class Bridge:
    def __init__(self, config, mode, interface='eth0', output=None):
        self.config, self.mode, self.interface = config, mode, interface
        self.output = output
        self.process = None
        self.state = None
        self.previous_state = None
        self.armed = False
        self.sequence = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(('127.0.0.1', config.deployment['state_port']))
            self.sock.setblocking(False)
        except OSError as exc:
            self.sock.close()
            if exc.errno == errno.EADDRINUSE:
                raise RuntimeError('运控反馈端口 127.0.0.1:%d 已被占用，请先关闭旧 Viser 或导航运控进程。'
                                   '只更换网页 --port 不能解决运控端口冲突。' % config.deployment['state_port']) from exc
            raise
        self.destination = ('127.0.0.1', config.deployment['bridge_port'])
        self.started = time.monotonic()

    def start(self, armed=False):
        if armed and self.mode == 'observe':
            raise RuntimeError('Observe mode cannot acquire motor control')
        self.stop_process()
        self.state = None
        self.previous_state = None
        self.sequence = 0
        while True:
            try:
                self.sock.recvfrom(p.STATE.size + 1)
            except BlockingIOError:
                break
        command = [str(DEPLOY / 'build/g1_control'), '--single-waist', '--prepare-full-body',
                   '--dex3-grasp', ','.join(format(v, '.9g') for v in self.config.hand_grasp.flat),
                   '--dex3-pd', ','.join(format(self.config.hand_pd[key], '.9g') for key in ('kp', 'kd'))]
        if self.config.hand_zero is not None:
            command += ['--dex3-zero-compensated', ','.join(format(v, '.9g') for v in self.config.hand_zero.flat)]
        if self.mode == 'sim':
            command += ['--simulate']
        else:
            command += ['--interface', self.interface]
        if armed:
            command += ['--control']
            if self.mode != 'sim':
                command += ['--release-mode']
        self.process = subprocess.Popen(command, stdout=self.output, stderr=subprocess.STDOUT)
        self.armed = armed
        self.started = time.monotonic()

    def read(self):
        for _ in range(128):
            try:
                data, sender = self.sock.recvfrom(p.STATE.size + 1)
            except BlockingIOError:
                break
            if sender != self.destination:
                continue
            state = p.State.decode(data)
            if state.config_token != self.config.token:
                raise RuntimeError('配置不一致，请重新编译并启动')
            if self.state is not None and state.session != self.state.session:
                raise RuntimeError('控制桥接意外重启，请退出并重新启动')
            if self.state is None or state.sequence >= self.state.sequence:
                if self.state is not None and state.received_ns > self.state.received_ns:
                    self.previous_state = self.state
                self.state = state
        if self.process is not None and self.process.poll() is not None:
            if self.state is None or self.state.stage not in (p.DAMPED, p.FAULT):
                raise RuntimeError('控制桥接退出，退出码 %s' % self.process.returncode)
        return self.state

    def send(self, request, q, profile=0, hand_pose=0, hand_recover=-1, hand_grasp=None, hand_live=False,
             upper_mode=0, upper_target_active=False):
        if self.mode == 'observe' or not self.armed or self.state is None:
            return
        self.sequence += 1
        self.sock.sendto(p.command(self.state, self.sequence, request, q, profile, hand_pose, hand_recover,
                                  hand_grasp, hand_live, upper_mode, upper_target_active), self.destination)

    def stop_process(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            # SIGTERM invokes the bridge's own one-second damping shutdown.
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.process = None

    def close(self):
        try:
            if self.armed and self.state is not None:
                self.send(p.STOP, self.state.applied_q)
        finally:
            self.stop_process()
            self.sock.close()
