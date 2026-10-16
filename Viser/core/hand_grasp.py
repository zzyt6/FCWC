"""Named Dex3 grasp angles and atomic edits of the existing upper-pose YAML."""
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml

HAND_JOINTS = ('thumb_0', 'thumb_1', 'thumb_2', 'middle_0', 'middle_1', 'index_0', 'index_1')
# Command limits match cpp/hand_core.hpp; feedback has a separate 0.02 rad margin.
HAND_LOWER = np.array([[-1.05, -.724, 0, -1.57, -1.75, -1.57, -1.75],
                       [-1.05, -1.05, -1.75, 0, 0, 0, 0]])
HAND_UPPER = np.array([[1.05, 1.05, 1.75, 0, 0, 0, 0],
                       [1.05, .742, 0, 1.57, 1.75, 1.57, 1.75]])


def validate_grasp(values, zero=None):
    raw = np.asarray(values)
    if raw.shape != (2, 7) or raw.dtype.kind not in 'fiu':
        raise ValueError('Dex3 抓握姿态必须包含 14 个数值角度')
    raw = raw.astype(float)
    if not np.isfinite(raw).all() or (raw < HAND_LOWER).any() or (raw > HAND_UPPER).any():
        raise ValueError('Dex3 抓握姿态存在非有限值或超出关节限位')
    if zero is not None:
        target = raw + zero
        if not np.isfinite(target).all() or (target < HAND_LOWER).any() or (target > HAND_UPPER).any():
            raise ValueError('Dex3 抓握角 + 补偿角超出关节限位')
    return raw.copy()


def _mapping(node):
    if not isinstance(node, yaml.MappingNode):
        raise ValueError('抓握 YAML 必须按关节名保存角度')
    result = {}
    for key, value in node.value:
        if not isinstance(key, yaml.ScalarNode) or key.value in result:
            raise ValueError('抓握 YAML 存在重复或无效字段')
        result[key.value] = value
    return result


def _grasp_nodes(text):
    root = _mapping(yaml.compose(text, Loader=yaml.SafeLoader))
    if 'hand_grasp' not in root:
        raise ValueError('YAML 缺少 hand_grasp')
    hands = _mapping(root['hand_grasp'])
    if set(hands) != {'left', 'right'}:
        raise ValueError('hand_grasp 必须包含 left 和 right')
    nodes = []
    for side in ('left', 'right'):
        joints = _mapping(hands[side])
        names = [f'{side}_hand_{joint}_joint' for joint in HAND_JOINTS]
        if set(joints) != set(names):
            raise ValueError(f'Dex3 {side} 必须包含完整的 7 个关节名')
        for name in names:
            node = joints[name]
            if (not isinstance(node, yaml.ScalarNode) or node.tag not in
                    ('tag:yaml.org,2002:float', 'tag:yaml.org,2002:int')
                    or not hands[side].start_mark.index <= node.start_mark.index < hands[side].end_mark.index):
                raise ValueError(f'{name} 必须直接填写数值角度（弧度）')
            nodes.append(node)
    if len({node.start_mark.index for node in nodes}) != 14:
        raise ValueError('抓握角度不能使用 YAML 别名')
    return nodes


def load_grasp(path):
    text = Path(path).read_text(encoding='utf-8')
    values = [yaml.safe_load(text[node.start_mark.index:node.end_mark.index]) for node in _grasp_nodes(text)]
    return validate_grasp(np.asarray(values).reshape(2, 7))


def save_grasp(path, values):
    """Replace only 14 scalar values, retaining body targets, comments and order."""
    raw = validate_grasp(values)
    path = Path(path)
    original = path.read_text(encoding='utf-8')
    replacements = [(node.start_mark.index, node.end_mark.index, repr(float(value)))
                    for node, value in zip(_grasp_nodes(original), raw.flat)]
    updated = original
    for start, end, value in sorted(replacements, reverse=True):
        updated = updated[:start] + value + updated[end:]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', delete=False) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), path.stat().st_mode & 0o777)
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        if path.read_text(encoding='utf-8') != original:
            raise ValueError('保存期间 YAML 已被其他程序修改，请重新加载后保存')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
