"""Latest-state selection preserves actual packet timing; no sockets or motors."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.backends import Bridge
from core.heading import DEFAULTS, validate_settings
from core.pelvis_imu import PelvisImu


class FeedbackTimingTests(unittest.TestCase):
    def setUp(self):
        self.bridge=Bridge.__new__(Bridge)
        self.bridge.config=SimpleNamespace(token=7)
        self.bridge.state=self.bridge.previous_state=self.bridge.process=None
        self.bridge.feedback_times=deque(maxlen=256)
        self.bridge.destination=('127.0.0.1',39000)
        self.bridge.sock=Mock()

    def read(self, packets):
        self.bridge.sock.recvfrom.side_effect=[(b'packet',self.bridge.destination) for _ in packets]+[BlockingIOError()]
        with patch('core.backends.p.State.decode',side_effect=packets):
            return self.bridge.read()

    @staticmethod
    def packet(sequence, stamp, flags=1, session=42):
        return SimpleNamespace(config_token=7,session=session,sequence=sequence,
                               received_ns=round(stamp*1e9),flags=flags)

    def test_latest_feedback_keeps_evidence_of_intermediate_imu_samples(self):
        imu=PelvisImu(validate_settings(DEFAULTS))
        first=self.read([self.packet(1,10.)])
        imu.update([1,0,0,0],[0]*3,(first.session,first.sequence),10.,10.03)
        packets=[self.packet(i+1,10.+i*.005) for i in range(1,25)]
        latest=self.read(packets)
        self.assertIs(latest,packets[-1])
        self.assertIs(self.bridge.previous_state,packets[-2])
        self.assertEqual(len(self.bridge.feedback_times),25)
        sample=imu.update([1,0,0,0],[0]*3,(latest.session,latest.sequence),10.12,10.121,
                          feedback_times=self.bridge.feedback_times)
        self.assertTrue(sample['imu_consumer_resumed'])
        self.assertEqual(sample['imu_health'],'warming_up')

    def test_real_stream_gap_is_reported_for_standing_recovery(self):
        imu=PelvisImu(validate_settings(DEFAULTS))
        self.read([self.packet(1,10.)])
        imu.update([1,0,0,0],[0]*3,(42,1),10.,10.)
        latest=self.read([self.packet(25,10.12)])
        sample=imu.update([1,0,0,0],[0]*3,(42,latest.sequence),10.12,10.121,
                          feedback_times=self.bridge.feedback_times)
        self.assertTrue(sample['imu_gap_detected'])
        self.assertEqual(sample['imu_health'],'sample_gap_recovering')

    def test_duplicate_backward_and_unhealthy_packets_cannot_refresh_continuity(self):
        first=self.read([self.packet(10,10.)])
        self.read([first,self.packet(9,9.99)])
        np.testing.assert_allclose(self.bridge.feedback_times,[10.])
        self.read([self.packet(11,10.01,flags=0)])
        self.assertFalse(self.bridge.feedback_times)
        self.read([self.packet(12,10.02)])
        np.testing.assert_allclose(self.bridge.feedback_times,[10.02])
        with self.assertRaisesRegex(RuntimeError,'控制桥接意外重启'):
            self.read([self.packet(13,10.03,session=43)])


if __name__=='__main__':
    unittest.main()
