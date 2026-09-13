"""Conservative edge cleanup and measurable dubbing timeline diagnostics."""
import os

import numpy as np

from tools.dubbing_settings import read_json, write_json_atomic


def trim_edge_silence(wav, sample_rate):
    """Remove only long, near-silent edges. Keep breaths and all internal pauses.

    The threshold is at most -66 dBFS and 60 dB below this clip's peak. Leave
    80 ms before the first activity and 120 ms after the last activity. Never
    trim an all-silent/very quiet clip or an edge shorter than 150 ms.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if not wav.size or sample_rate <= 0:
        return wav
    peak = float(np.max(np.abs(wav)))
    if not np.isfinite(wav).all():
        raise ValueError('配音音檔含有無效樣本')
    if peak <= 0.0005:
        return wav
    active = np.flatnonzero(np.abs(wav) > min(0.0005, peak * 0.001))
    if not active.size:
        return wav
    first = max(0, int(active[0]) - round(0.08 * sample_rate))
    last = min(wav.size, int(active[-1]) + 1 + round(0.12 * sample_rate))
    if first < round(0.15 * sample_rate):
        first = 0
    if wav.size - last < round(0.15 * sample_rate):
        last = wav.size
    return wav[first:last]


def timing_report(transcript):
    rows = []
    for i, line in enumerate(transcript):
        timing = line.get('dub_timing')
        if not timing:
            continue
        rows.append({'index': i, 'speaker': line.get('speaker', ''),
                     'text': line.get('translation', ''), **timing})
    spoken = [row for row in rows if not row.get('skipped')]
    delays = [row['delay_seconds'] for row in spoken]
    return {
        'version': 1, 'line_count': len(rows),
        'delayed_lines': sum(delay >= 0.8 for delay in delays),
        'max_delay_seconds': round(max(delays, default=0), 3),
        'trimmed_silence_seconds': round(sum(row.get('trimmed_seconds', 0) for row in rows), 3),
        'rows': rows,
    }


def write_timing_report(folder, transcript):
    report = timing_report(transcript)
    write_json_atomic(os.path.join(folder, 'dubbing_timing.json'), report)
    return report


def load_timing_ui(folder):
    from tools.speaker_edit import episode_folder
    folder = episode_folder(folder)
    if not folder:
        return '請先選擇已配音的影片資料夾。', []
    # Read active lines, not another language's stale diagnostic file.
    lines = read_json(os.path.join(folder, 'translation.json'), [])
    report = timing_report(lines)
    if not report['rows']:
        return '尚無時間軸診斷，請先用新版執行配音。', []
    summary = (
        f"已檢查 {report['line_count']} 句；延遲 ≥ 0.8 秒：{report['delayed_lines']} 句；"
        f"最大延遲：{report['max_delay_seconds']:.2f} 秒；"
        f"已移除首尾多餘靜音：{report['trimmed_silence_seconds']:.2f} 秒。\n\n"
        '依延遲由大到小排列。請到「講者／譯文」修改對應句；語速固定仍可能因譯文過長而延遲。'
    )
    table = []
    for row in sorted(report['rows'], key=lambda row: (-row['delay_seconds'], row['index'])):
        table.append([row['index'], row['speaker'], row['orig_start'], row['start'],
                      row['slot_seconds'], row['duration_seconds'], row['delay_seconds'],
                      '略過' if row.get('skipped') else ('需檢查' if row['delay_seconds'] >= 0.8 else '正常'),
                      row['text']])
    return summary, table
