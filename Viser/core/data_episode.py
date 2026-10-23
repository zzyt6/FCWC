"""UDP control shared by the push-cart data recorders (standard library only).

The coordinator in the control process (core/pushcart_data.py) owns the episode:
push-cart enabled -> disabled. Ten times per second it sends the current episode
to each recorder on loopback. Every recorder keeps its own samples whose
host-monotonic timestamp lies in [start - pre, stop + post], so all files of one
episode cover the same time window regardless of control-message latency.

Line format (ASCII, one datagram):
    PUSHCART_DATA 1 <seq> <directory|-> <start_s> <stop_s|nan> <pre_s> <post_s>
Reply from each recorder to the sender:
    PUSHCART_DATA_ACK 1 <name> <seq> <directory|-> <rows> <dropped>
"""
from collections import deque
import json
import math
from pathlib import Path
import socket
import time

PRE_S = 1.0
POST_S = 1.0
# The coordinator keeps announcing a stopped episode this long after stop+post,
# so queued samples up to stop+post are still written.
CLOSE_MARGIN_S = 1.0
# A recorder closes its episode when the coordinator stops talking.
SILENCE_S = 3.0
CONTROL_PERIOD_S = 0.1
CONTROL_PORTS = dict(camera=39011, ros=39012)


def encode(seq, directory, start=math.nan, stop=math.nan, pre=PRE_S, post=POST_S):
    directory = '-' if directory is None else str(directory)
    if any(c.isspace() for c in directory):
        raise ValueError('Episode directory must not contain whitespace')
    return ('PUSHCART_DATA 1 %d %s %.6f %.6f %.3f %.3f'
            % (seq, directory, start, stop, pre, post)).encode('ascii')


def decode(data):
    """Return dict(seq, directory|None, start, stop, pre, post) or None if invalid."""
    try:
        magic, version, seq, directory, start, stop, pre, post = data.decode('ascii').split()
        if magic != 'PUSHCART_DATA' or version != '1':
            return None
        start, stop, pre, post = map(float, (start, stop, pre, post))
        if directory == '-':
            return dict(seq=int(seq), directory=None)
        if (not Path(directory).is_absolute() or not math.isfinite(start)
                or not 0 <= pre <= 10 or not 0 <= post <= 10 or math.isinf(stop)):
            return None
        return dict(seq=int(seq), directory=directory, start=start,
                    stop=math.inf if math.isnan(stop) else stop, pre=pre, post=post)
    except (ValueError, UnicodeDecodeError):
        return None


def decode_ack(data):
    try:
        magic, version, name, seq, directory, rows, dropped = data.decode('ascii').split()
        if magic != 'PUSHCART_DATA_ACK' or version != '1':
            return None
        return dict(name=name, seq=int(seq), directory=None if directory == '-' else directory,
                    rows=int(rows), dropped=int(dropped))
    except (ValueError, UnicodeDecodeError):
        return None


class EpisodeRecorder:
    """Recorder side: pre-roll buffer, sample window and control socket.

    A subclass implements open_files(directory), write(t, sample) and
    close_files() -> dict. feed() and service() must run on one thread.
    """
    def __init__(self, name, port, *, clock=time.monotonic):
        self.name = name
        self.clock = clock
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(('127.0.0.1', int(port)))
            self.sock.setblocking(False)
        except OSError:
            self.sock.close()
            raise
        self.episode = None
        self.preroll = deque()
        self.last_control = -math.inf
        self.dropped = 0  # samples lost before reaching feed(); subclasses count them

    def feed(self, t, sample):
        episode = self.episode
        if episode is not None and t >= episode['start'] - episode['pre']:
            if t <= episode['stop'] + episode['post']:
                self._write(t, sample)
            return
        self.preroll.append((t, sample))
        while self.preroll and self.preroll[0][0] < t - PRE_S - .5:
            self.preroll.popleft()

    def service(self, now=None):
        now = self.clock() if now is None else now
        for _ in range(64):
            try:
                data, sender = self.sock.recvfrom(512)
            except BlockingIOError:
                break
            except OSError:
                break
            message = decode(data)
            if message is None or sender[0] != '127.0.0.1':
                continue
            self.last_control = now
            self._apply(message)
            episode = self.episode
            reply = 'PUSHCART_DATA_ACK 1 %s %d %s %d %d' % (
                self.name, message['seq'], episode['directory'] if episode else '-',
                episode['rows'] if episode else 0, self.dropped - episode['dropped_at_open'] if episode else 0)
            try:
                self.sock.sendto(reply.encode('ascii'), sender)
            except OSError:
                pass
        if self.episode is not None and now - self.last_control > SILENCE_S:
            self._close('control_lost')

    def close(self):
        if self.episode is not None:
            self._close('shutdown')
        self.sock.close()

    def _apply(self, message):
        directory = message['directory']
        if directory is None:
            if self.episode is not None:
                self._close('stopped')
            return
        if self.episode is not None and self.episode['directory'] != directory:
            self._close('superseded')
        if self.episode is None:
            self.episode = dict(message, rows=0, first_t=None, last_t=None,
                                dropped_at_open=self.dropped, opened_monotonic=self.clock())
            try:
                self.open_files(Path(directory))
            except Exception as exc:
                print('PUSHCART DATA %s warning: cannot open %s: %s' % (self.name, directory, exc), flush=True)
                self.episode = None
                return
            buffered, self.preroll = self.preroll, deque()
            for t, sample in buffered:
                self.feed(t, sample)
        else:
            # A program exit shortens the post-roll of the running episode.
            self.episode['stop'], self.episode['post'] = message['stop'], message['post']

    def _write(self, t, sample):
        episode = self.episode
        self.write(t, sample)
        episode['rows'] += 1
        if episode['first_t'] is None:
            episode['first_t'] = t
        episode['last_t'] = t

    def _close(self, reason):
        episode, self.episode = self.episode, None
        try:
            extra = self.close_files() or {}
        except Exception as exc:
            extra = dict(close_error=repr(exc))
        summary = dict(stream=self.name, close_reason=reason, rows=episode['rows'],
                       dropped_before_write=self.dropped - episode['dropped_at_open'],
                       first_t=episode['first_t'], last_t=episode['last_t'],
                       window=dict(start=episode['start'],
                                   stop=None if math.isinf(episode['stop']) else episode['stop'],
                                   pre=episode['pre'], post=episode['post']), **extra)
        try:
            (Path(episode['directory']) / (self.name + '_summary.json')).write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
        except OSError as exc:
            print('PUSHCART DATA %s warning: summary not written: %s' % (self.name, exc), flush=True)
        print('PUSHCART DATA %s closed %s rows=%d reason=%s'
              % (self.name, episode['directory'], episode['rows'], reason), flush=True)

    def open_files(self, directory):
        raise NotImplementedError

    def write(self, t, sample):
        raise NotImplementedError

    def close_files(self):
        raise NotImplementedError
