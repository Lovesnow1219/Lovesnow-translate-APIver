# -*- coding: utf-8 -*-
"""Flag dubbed-line problems after translation. Never writes translation.json."""
import json
import os
import re
from datetime import datetime, timezone

from loguru import logger

from tools.target_language import is_asr_junk, is_chinese_target, load_dub_meta, translation_language
from tools.translation_quality import HAN_RE, JA_CN_LEFT_RE, KANA_RE, leftover_chinese_reason
from tools.vocal_particles import is_particle_card, particle_translation

REVIEW_NAME = 'translation_review.json'
_BATCH = 10
_NAME_CN_RE = re.compile(r'[威薇维]拉|本小姐')
_JA_KANA_NUM = re.compile(r'[零一二三四五六七八九十百千万億兆]+')
_DIGIT_RE = re.compile(r'[0-9０-９]')
_STUMP_SRC = {'做', '所以', '究。', '究', '归了', '归了。', '侯'}
_GLOSS_SKIP_RE = re.compile(r'世界|现代|現代|战争|戰爭|游戏|遊戲|银行|銀行|货币|貨幣|金币|金幣|空气|空氣|卡$')
_SEVERITY_RANK = {'high': 0, 'mid': 1, 'low': 2}


def review_path(folder):
    return os.path.join(folder, REVIEW_NAME)


def load_review(folder):
    path = review_path(folder)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def review_table_rows(folder):
    findings = (load_review(folder).get('findings') or [])
    rows = []
    for item in findings:
        rows.append([
            item.get('index'),
            item.get('start'),
            item.get('severity') or '',
            item.get('issue') or '',
            item.get('suggest') or '',
        ])
    return rows


def review_status_line(folder):
    report = load_review(folder)
    findings = report.get('findings') or []
    if not findings:
        if report:
            return '審稿：沒有標出異常（譯文未改）'
        return '審稿：尚未跑過'
    counts = {'high': 0, 'mid': 0, 'low': 0}
    for item in findings:
        key = item.get('severity') if item.get('severity') in counts else 'mid'
        counts[key] += 1
    return (
        f'審稿標出 {len(findings)} 句（高 {counts["high"]}／中 {counts["mid"]}／低 {counts["low"]}）。'
        '譯文未改，請對照講者表後自行儲存。'
    )


def review_folder(folder, target_language=None, method='OpenAI'):
    """Write translation_review.json. Leaves translation.json untouched."""
    path = os.path.join(folder, 'translation.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'找不到 translation.json：{folder}')
    with open(path, 'r', encoding='utf-8') as handle:
        transcript = json.load(handle)
    if not isinstance(transcript, list):
        raise ValueError('translation.json 格式不正確')
    language = translation_language(
        target_language or load_dub_meta(folder).get('translation') or 'English'
    )
    summary = _load_summary(folder)
    rules = _rule_findings(transcript, language, summary)
    suspects = _suspect_indices(transcript, language, rules)
    ai_findings = []
    if not is_chinese_target(language) and method in {'OpenAI', 'LLM', '阿里云-通义千问', 'Ernie', 'Ollama'}:
        ai_findings = _ai_review(transcript, language, summary, suspects, method)
    findings = _merge_findings(transcript, rules, ai_findings)
    report = {
        'language': language,
        'reviewed_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'line_count': len(transcript),
        'suspect_count': len(suspects),
        'applied': False,
        'findings': findings,
    }
    with open(review_path(folder), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    logger.info(f'審稿寫入 {len(findings)} 條、未改譯文：{folder}')
    return report


def _load_summary(folder):
    path = os.path.join(folder, 'summary.json')
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _glossary_pairs(summary):
    pairs = []
    raw = (summary or {}).get('glossary') or ''
    for chunk in re.split(r'[;\n]+', raw):
        if '=' not in chunk:
            continue
        left, right = chunk.split('=', 1)
        src = left.strip()
        dst = right.strip()
        if src and dst and _is_name_gloss(src, dst):
            pairs.append((src, dst))
    return pairs


def _is_name_gloss(src, dst):
    if len(src) < 2 or _GLOSS_SKIP_RE.search(src):
        return False
    if re.fullmatch(r'[\u30a0-\u30ffー]+', dst or ''):
        return True
    return bool(re.fullmatch(r"[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+)?", dst or ''))


def _line_start(line):
    try:
        return round(float(line.get('orig_start', line.get('start') or 0)), 2)
    except (TypeError, ValueError):
        return 0.0


def _compact_src(text):
    return ''.join(ch for ch in (text or '') if ch.strip() and ch not in '，,。！？!?、…')


def _rule_findings(transcript, language, summary):
    glossary = _glossary_pairs(summary)
    out = []
    for i, line in enumerate(transcript or []):
        src = (line.get('text') or '').strip()
        dub = (line.get('translation') or '').strip()
        issues = []
        suggest = ''
        leftover = leftover_chinese_reason(dub, language)
        if leftover:
            issues.append('譯文殘留中文')
        if language == 'Japanese' and JA_CN_LEFT_RE.search(dub):
            issues.append('日文裡出現簡體語氣字')
        if language != '简体中文' and _NAME_CN_RE.search(dub):
            issues.append('譯文還留著中文人名或「本小姐」')
        expected = particle_translation(src, language)
        if expected and dub and expected.rstrip('。. ') not in dub:
            issues.append(f'語氣詞應為 {expected.rstrip()}')
            suggest = expected
        if '本小姐' in src:
            if language == 'Japanese' and 'この私' not in dub and '私' not in dub:
                issues.append('「本小姐」應譯成這個說話的人（この私）')
            if language == 'Vietnamese' and 'tôi' not in dub.lower() and 'em' not in dub.lower():
                issues.append('「本小姐」應譯成說話者自己（tôi）')
        if any(mark in src for mark in ('好看，买', '好看，買', '好看买')):
            if language == 'Japanese' and '買う' not in dub and '買' not in dub:
                issues.append('「好看，买」應保留買的動作')
            if language == 'Vietnamese' and 'mua' not in dub.lower():
                issues.append('「好看，买」應保留 mua')
        if '横着走' in src or '横着走走' in src:
            if '横向き' in dub or '横向きで' in dub:
                issues.append('「橫著走」是神氣走路，不是側身走')
        if '快乐水' in src or '快樂水' in src:
            if '幸せの水' in dub or 'hạnh phúc' in dub.lower():
                issues.append('「快樂水」是可樂／汽水，不要直譯成幸福的水')
        if _compact_src(src) in _STUMP_SRC or (len(_compact_src(src)) <= 2 and src and not is_particle_card(src)):
            if dub and len(dub) > 8:
                issues.append('原文像辨識殘句，譯文卻寫成完整句')
        if src.rstrip().endswith(('，', ',', '、')) and dub.rstrip().endswith(('。', '.', '！', '!')):
            issues.append('原文被切斷，譯文卻收成完整句，可能跟下一句對不起來')
        if is_asr_junk(src):
            issues.append('原文像辨識幻聽')
        if language == 'Japanese' and len(dub) >= 18 and KANA_RE.search(dub) and not HAN_RE.search(dub):
            issues.append('整句幾乎沒有漢字，可能是品質檢查逼成全假名')
        if _DIGIT_RE.search(src) and language == 'Japanese' and _JA_KANA_NUM.search(dub) and not _DIGIT_RE.search(dub):
            if any(token in src for token in ('LV', 'lv', '%', '/', '0/10')):
                issues.append('原文有阿拉伯數字或 LV／％，譯文寫成漢字數字')
        for cn, dst in glossary:
            if cn in src and dst and dst not in dub and cn not in ('篝火', '金币', '金幣'):
                if len(cn) >= 2:
                    issues.append(f'人名「{cn}」大綱是 {dst}，譯文沒對上')
                    break
        if not issues:
            continue
        out.append({
            'index': i,
            'start': _line_start(line),
            'speaker': line.get('speaker') or '',
            'source': src,
            'translation': dub,
            'severity': 'high' if any(key in ''.join(issues) for key in ('殘留中文', '中文人名', '側身', '快樂水', '本小姐')) else 'mid',
            'issue': '；'.join(issues),
            'suggest': suggest,
            'via': 'rule',
        })
    return out


def _suspect_indices(transcript, language, rules):
    marked = {item['index'] for item in rules}
    for i, line in enumerate(transcript or []):
        if i in marked:
            continue
        src = line.get('text') or ''
        dub = line.get('translation') or ''
        if not dub:
            marked.add(i)
            continue
        if leftover_chinese_reason(dub, language) or _NAME_CN_RE.search(dub):
            marked.add(i)
        if src.rstrip().endswith(('，', ',', '、', '…')):
            marked.add(i)
        if is_particle_card(src) or '本小姐' in src:
            marked.add(i)
        if any(token in src for token in ('横着', '快乐水', '快樂水', '好看', '滚蛋', '侯', '归了')):
            marked.add(i)
    return sorted(marked)


def _ai_review(transcript, language, summary, suspects, method):
    if not suspects:
        return []
    from tools.translation_backends import llm_translate

    system = (
        'You are a dubbing QC editor, not the translator. '
        'The operator reads Traditional Chinese. '
        'Flag only real problems in the dubbed line. '
        'Do not treat Japanese kanji as leftover Chinese. '
        'Do not rewrite fine lines. '
        'Do not invent plot. '
        'If the Chinese source is a broken ASR stump, say so; do not praise a fluent dub of junk. '
        'Particles: 呵=ふふ/Hề, 哼=ふん/Hừ, 嘿嘿=へへ/He he, 嗯=ん/Ừ. '
        '本小姐 is the speaker (この私 / tôi), not a name. '
        '横着走 is swagger, not walking sideways. '
        '快乐水 is soda/cola. '
        'Keep Arabic digits for LV99, 30, 640, 0/10, %. '
        'Output JSON only: {"findings":[{"index":0,"severity":"high|mid|low","issue":"繁體中文","suggest":"replacement or empty"}]}'
    )
    glossary = (summary or {}).get('glossary') or ''
    title = (summary or {}).get('title') or ''
    findings = []
    for start in range(0, len(suspects), _BATCH):
        chunk = suspects[start:start + _BATCH]
        payload = [f'Title: {title}', f'Language: {language}', f'Glossary: {glossary}', '']
        for index in chunk:
            line = transcript[index]
            prev_src = transcript[index - 1].get('text') if index else ''
            next_src = transcript[index + 1].get('text') if index + 1 < len(transcript) else ''
            payload.append(
                f'#{index} {line.get("speaker") or ""} {_line_start(line)}s\n'
                f'SRC: {line.get("text") or ""}\n'
                f'DUB: {line.get("translation") or ""}\n'
                f'PREV: {prev_src}\n'
                f'NEXT: {next_src}'
            )
        try:
            raw = llm_translate(method, [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': '\n\n'.join(payload)},
            ])
        except Exception as exc:
            logger.warning(f'審稿模型失敗：{exc}')
            continue
        findings.extend(_parse_ai_findings(raw, chunk))
    return findings


def _parse_ai_findings(raw, allowed):
    text = (raw or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?', '', text).rstrip('`').strip()
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except Exception:
        logger.warning('審稿回傳不是 JSON，略過這批')
        return []
    rows = data.get('findings') if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    allowed_set = set(allowed)
    out = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get('index'))
        except (TypeError, ValueError):
            continue
        if index not in allowed_set:
            continue
        issue = str(item.get('issue') or '').strip()
        if not issue:
            continue
        severity = str(item.get('severity') or 'mid').lower()
        if severity not in _SEVERITY_RANK:
            severity = 'mid'
        out.append({
            'index': index,
            'severity': severity,
            'issue': issue,
            'suggest': str(item.get('suggest') or '').strip(),
            'via': 'ai',
        })
    return out


def _merge_findings(transcript, rules, ai_findings):
    by_index = {}
    for item in rules:
        by_index[item['index']] = dict(item)
    for item in ai_findings:
        index = item['index']
        line = transcript[index] if 0 <= index < len(transcript) else {}
        current = by_index.get(index)
        if current is None:
            by_index[index] = {
                'index': index,
                'start': _line_start(line),
                'speaker': line.get('speaker') or '',
                'source': line.get('text') or '',
                'translation': line.get('translation') or '',
                'severity': item.get('severity') or 'mid',
                'issue': item.get('issue') or '',
                'suggest': item.get('suggest') or '',
                'via': 'ai',
            }
            continue
        issues = [current.get('issue') or '', item.get('issue') or '']
        current['issue'] = '；'.join(part for part in issues if part)
        if _SEVERITY_RANK.get(item.get('severity'), 9) < _SEVERITY_RANK.get(current.get('severity'), 9):
            current['severity'] = item.get('severity')
        if item.get('suggest') and not current.get('suggest'):
            current['suggest'] = item['suggest']
        elif item.get('suggest') and item.get('suggest') != current.get('suggest'):
            current['suggest'] = item['suggest']
        current['via'] = 'ai+rule' if current.get('via') == 'rule' else current.get('via') or 'ai'
    return sorted(by_index.values(), key=lambda row: (_SEVERITY_RANK.get(row.get('severity'), 9), row.get('index', 0)))
