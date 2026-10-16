"""Compact joint command/error monitor and operator warnings."""
import html
import math

from core.joint_tracking import NAMES, GROUPS, status, joint_label
from core.joint_monitor import limits, out_of_range, live_warnings, WarningBoard, BridgeLogWatcher, load_monitor_settings

PANEL_WIDTH = 440
# Pair left/right joints; the single waist row has an empty right half.
SECTIONS = (('腿部', range(0, 6), range(6, 12)),
            ('腰部', (12,), (None,)),
            ('手臂', range(13, 20), range(20, 27)),
            ('Dex3', range(27, 34), range(34, 41)))

# Scroll on Viser's stable HTML host, not the replaced inner markup, so live
# updates preserve the operator's scroll position. Selectors match only us.
STYLE = '''<style>
.jm{overflow:visible}
.jm table{width:100%;border-collapse:collapse;font-size:11px;font-variant-numeric:tabular-nums}
.jm th{position:sticky;top:0;z-index:1;background:var(--mantine-color-default);font-weight:600;text-align:right;padding:2px 3px}
.jm th:first-child,.jm td:first-child{text-align:left}
.jm td{padding:1px 3px;line-height:15px;text-align:right;border-bottom:1px solid var(--mantine-color-default-border);white-space:nowrap}
.jm td.n{text-align:left}
.jm td:nth-child(4),.jm th:nth-child(4){border-left:1px solid var(--mantine-color-default-border)}
.jm td.warn{background:#fff3bf;color:#854d0e;font-weight:600}
.jm tr.g td{background:var(--mantine-color-teal-light);color:var(--mantine-color-teal-light-color);font-weight:600;text-align:left}
.jm .stale{color:var(--mantine-color-dimmed)}
.jm td.bad{background:#ffe3e3;color:#b91c1c;font-weight:600}
.jm .hint{font-size:11px;color:var(--mantine-color-dimmed);margin:2px 0 6px}
div:has(> .jw){max-height:160px;overflow:auto;overscroll-behavior:contain}
.jw{font-size:12px;overflow-wrap:anywhere}
.jw div{padding:3px 6px;margin:2px 0;border-radius:4px;border-left:3px solid}
.jw .error{background:#fff5f5;border-color:#e03131;color:#c92a2a}
.jw .warning{background:#fff9db;border-color:#f08c00;color:#a35f00}
.jw .old{opacity:.55}
.jw .ok{background:#ebfbee;border-color:#2f9e44;color:#2b8a3e}
.jw .pending{background:var(--mantine-color-default);border-color:#868e96;color:var(--mantine-color-dimmed)}
</style>'''


def angle(value):
    return '—' if value is None or not math.isfinite(value) else '%+.1f' % math.degrees(value)


def joint_table(comparison, bad, now, error_threshold_deg=5.):
    now_ns = int(now * 1e9)

    def cells(index):
        if index is None:
            return '<td colspan="3"></td>'
        name = NAMES[index]
        valid, reason = status(comparison, index, now_ns) if comparison else (False, '等待关节指令与反馈')
        command, actual = (comparison['command_rad'][index], comparison['actual_rad'][index]) if comparison else (None, None)
        error = command - actual if valid else None
        warning = valid and abs(math.degrees(error)) > error_threshold_deg
        flag = 'bad' if index in bad else 'warn' if warning else ''
        group = GROUPS[index]
        sent = comparison['sent_ns'][group] if comparison else 0
        fresh = sent and 0 <= now_ns-sent <= comparison['timeout_s'][group]*1e9
        title = html.escape('%s · %s' % (name, '位置超限' if index in bad else reason))
        return ('<td class="n %s" title="%s">%s%s</td>'
                '<td class="%s">%s</td><td class="%s">%s</td>' % (
                    flag, title, html.escape(joint_label(name)), ' !' if index in bad else '',
                    flag or ('' if fresh else 'stale'), angle(command),
                    flag or ('' if valid else 'stale'), angle(error)))

    rows = []
    for title, left, right in SECTIONS:
        rows.append('<tr class="g"><td colspan="6">%s</td></tr>' % title)
        rows.extend('<tr>%s%s</tr>' % (cells(l), cells(r)) for l, r in zip(left, right))
    heading = '<th scope="col">关节</th><th scope="col">下发 °</th><th scope="col">误差 °</th>'
    return ('<div class="jm" role="region" aria-label="全部关节角度">'
            '<div class="hint">%s误差 = 下发 − 实际 · 黄色 |误差| &gt; %g° · 红色为位置超限</div>'
            '<table><thead><tr>%s%s</tr></thead><tbody>%s</tbody></table></div>' % (
                '<b>合成反馈</b> · ' if comparison and comparison.get('simulation') else '',
                error_threshold_deg, heading, heading, ''.join(rows)))


def warning_box(rows, ready=True):
    if not rows:
        return ('<div class="jw"><div class="ok">无告警</div></div>' if ready else
                '<div class="jw"><div class="pending">等待机器人反馈…</div></div>')
    return '<div class="jw">%s</div>' % ''.join(
        '<div class="%s%s">%s%s</div>' % (level, '' if live else ' old',
                                             '' if live else '刚结束 · ', html.escape(text))
        for level, text, live, _ in rows)


class JointMonitorPanel:
    """One left-docked panel per server; content is shared by every browser."""

    def __init__(self, server, controller):
        import viser
        self.control = controller
        self.lower, self.upper = limits(controller.config)
        self.board = WarningBoard()
        bridge_log = controller.recorder.path / 'bridge.log' if controller.recorder else None
        self.bridge = BridgeLogWatcher(bridge_log, controller.cfg['max_temperature_c'])
        self.settings = load_monitor_settings()
        self.panel = server.gui.add_panel()
        with self.panel.add_tab('关节监视', icon=viser.Icon.ROBOT):
            server.gui.add_html(STYLE)
            with server.gui.add_folder('关节角度', expand_by_default=True):
                self.table = server.gui.add_html(joint_table(None, set(), 0., self.settings['error_threshold_deg']))
            with server.gui.add_folder('告警信息', expand_by_default=True):
                self.warnings = server.gui.add_html(warning_box([], ready=False))

        self.panel.dock_left()
        self.panel.set_width(PANEL_WIDTH)
        self.last_html = (None, None)
        self.last_update = -math.inf

    def update(self, data, now):
        if now-self.last_update < .2:
            return
        self.last_update = now
        comparison = data.get('joint_comparison')
        bad = set(out_of_range(comparison, self.lower, self.upper))
        self.bridge.poll()
        current = live_warnings(data, comparison, self.lower, self.upper, self.control.cfg['state_timeout_s'], now=now)
        current += [warning for warning in self.bridge.warnings() if warning not in current]
        rows = self.board.update(current)
        table = joint_table(comparison, bad, now, self.settings['error_threshold_deg'])
        box = warning_box(rows, ready=data.get('state') is not None)
        if table != self.last_html[0]:
            self.table.content = table
        if box != self.last_html[1]:
            self.warnings.content = box
        self.last_html = (table, box)

