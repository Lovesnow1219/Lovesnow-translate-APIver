import json

import pytest

from tools.source_subtitles import apply_evidence, sample_times, MAX_FRAMES
from tools.translation_versions import write_translation, write_review_report


def test_repeated_substrings_require_explicit_original_occurrence():
    source = [{'text': '向左，向左，然后向右', 'start': 0, 'end': 3}]
    item = {'index': 0, 'evidence': [{'frame': 0, 'text': '向右，向左，然后向右'}],
            'edits': [{'from': '向左', 'to': '向右'}]}
    assert apply_evidence(source, [item], {0: 0}) == (source, [])
    item['edits'][0]['occurrence'] = 1
    result, changes = apply_evidence(source, [item], {0: 0})
    assert result[0]['text'] == '向右，向左，然后向右'
    assert changes[0]['edits'][0]['occurrence'] == 1
    # Both indices refer to the unmodified original even when lengths change.
    item['evidence'][0]['text'] = '向前边，向右边，然后向右'
    item['edits'] = [{'from': '向左', 'to': '向前边', 'occurrence': 1},
                     {'from': '向左', 'to': '向右边', 'occurrence': 2}]
    assert apply_evidence(source, [item], {0: 0})[0][0]['text'] == '向前边，向右边，然后向右'


def test_partial_caption_cannot_delete_unseen_clause_or_append_neighbor():
    source = [{'text': '请带雨伞和外套，今天下雨', 'start': 0, 'end': 3}]
    item = {'index': 0, 'evidence': [{'frame': 0, 'text': '请带雨伞'},
                                  {'frame': 1, 'text': '今天下雨明天降温'}],
            'edits': [{'from': '请带雨伞和外套', 'to': '请带雨伞'},
                      {'from': '今天下雨', 'to': '今天下雨明天降温'}]}
    assert apply_evidence(source, [item], {0: 0, 1: 0}) == (source, [])
    item['evidence'] = [{'frame': 0, 'text': '请带雨伞'}, {'frame': 1, 'text': '请带雨伞'}]
    item['edits'] = [{'from': source[0]['text'], 'to': '请带雨伞'}]
    assert apply_evidence(source, [item], {0: 0, 1: 0}) == (source, [])


def test_sampling_allocates_more_evidence_to_multi_caption_cards():
    lines = [{'start': i * 2, 'end': i * 2 + .5} for i in range(12)]
    lines.append({'start': 30, 'end': 39})
    plan = sample_times(lines, list(range(len(lines))), 40)
    assert len(plan[12]) > len(plan[0])
    assert max(b - a for a, b in zip(plan[12], plan[12][1:])) <= 1.3
    assert sum(map(len, plan.values())) <= MAX_FRAMES


def test_unchanged_caption_evidence_preserves_chronological_turn_boundaries():
    from tools.source_subtitles import apply_observations
    from tools.translation_bible import _transcript_for_bible
    source = [{'text': '你来了小李先坐下', 'start': 0, 'end': 3, 'speaker': 'A'}]
    observations = [{'index': 0, 'evidence': [
        {'frame': 3, 'text': '小李先坐下'}, {'frame': 8, 'text': 'Unrelated card'}]},
        {'index': 0, 'evidence': [{'frame': 1, 'text': '你来了'}, {'frame': 2, 'text': '你来了'}]}]
    result, accepted = apply_observations(source, observations, {1: 0, 2: 0, 3: 0, 8: 1})
    assert result[0]['text'] == source[0]['text'] and result[0]['speaker'] == 'A'
    assert result[0]['source_subtitle_evidence'] == ['你来了', '小李先坐下']
    assert len(accepted) == 1 and 'source_subtitle_evidence' not in source[0]
    assert '["你来了", "小李先坐下"]' in _transcript_for_bible(result)


def test_evidence_replay_requires_identical_source_and_video():
    from tools.source_subtitles import replay_observations
    source = [{'text': '你好，请坐', 'start': 0, 'end': 2, 'speaker': 'A'}]
    report = {'version': 6, 'completed': True, 'video': 'same', 'input_transcript': source,
              'corrections': [], 'observations': [
                  {'index': 0, 'evidence': [{'frame': 0, 'text': '你好'}]},
                  {'index': 0, 'evidence': [{'frame': 1, 'text': '请坐'}]}]}
    result, _ = replay_observations(source, report, 'same')
    assert result[0]['source_subtitle_evidence'] == ['你好', '请坐']
    assert replay_observations(source, report, 'different') is None
    changed = [{**source[0], 'text': '用户编辑过的原文'}]
    assert replay_observations(changed, report, 'same') is None


def test_word_estimate_cannot_reject_a_complete_faithful_line():
    from tools.translation_quality import valid_translation
    text = "I promised to help, but I can't stay until tomorrow."
    ok, result = valid_translation('我答应过帮忙，但我不能留到明天。', text, 'English', duration=1)
    assert ok and result == text


def prepare_review(folder):
    rows = [{'text': '明天不上班', 'translation': 'Work tomorrow.', 'speaker': 'A',
             'start': 0, 'end': 1, 'orig_start': 0, 'orig_end': 1}]
    write_translation(str(folder), rows, 'English')
    write_review_report(str(folder), 'English', {'language': 'English', 'findings': [
        {'index': 0, 'source': rows[0]['text'], 'translation': rows[0]['translation'],
         'suggest': "You don't have to work tomorrow.", 'severity': 'high', 'issue': '語意反轉'}]})


@pytest.mark.parametrize('response,accepted', [
    ('{"decisions":[{"index":0,"approved":true,"reason":"保留否定"}]}', True),
    ('{"decisions":[{"index":0,"approved":false,"reason":"無法確認"}]}', False),
    ('{"decisions":[]}', False),
    ('{"decisions":[{"index":0,"approved":"true","reason":"格式錯誤"}]}', False),
    ('{"decisions":[{"index":1,"approved":true,"reason":"錯句"}]}', False),
])
def test_automatic_rewrite_requires_explicit_matching_approval(tmp_path, monkeypatch, response, accepted):
    from tools.rewrite_validation import validate_rewrites
    prepare_review(tmp_path)
    monkeypatch.setattr('tools.translation_backends.llm_translate', lambda *a, **k: response)
    assert validate_rewrites(str(tmp_path), 'English', ['0']) == (['0'] if accepted else [])
    assert json.loads((tmp_path/'translation.json').read_text())[0]['translation'] == 'Work tomorrow.'


def test_failed_validation_keeps_original_and_cannot_stamp_as_completed(tmp_path, monkeypatch):
    from tools.rewrite_validation import validate_rewrites
    from tools.review_state import failures
    from tools.translation_review import stamp_after_dub, after_dub_already_done
    prepare_review(tmp_path)
    def fail(*a, **k):
        raise TimeoutError('Provider unavailable')
    monkeypatch.setattr('tools.translation_backends.llm_translate', fail)
    assert validate_rewrites(str(tmp_path), 'English', ['0']) == []
    assert failures(str(tmp_path), 'English')['自動改譯覆核'] == 'TimeoutError'
    stamp_after_dub(str(tmp_path), 'English')
    assert not after_dub_already_done(str(tmp_path), 'English')
    # A later review with no applicable rewrite must not retain a stale failure.
    write_review_report(str(tmp_path), 'English', {'language': 'English', 'findings': []})
    assert validate_rewrites(str(tmp_path), 'English', []) == []
    assert not failures(str(tmp_path), 'English')


def test_applying_approved_suggestion_preserves_exact_wording(tmp_path, monkeypatch):
    from tools.translation_review import apply_selected_review
    prepare_review(tmp_path)
    def unexpected(*a, **k):
        pytest.fail('Applying a reviewed suggestion must not call a second writer')
    monkeypatch.setattr('tools.translation.tighten_overlong_lines', unexpected)
    _, applied, _, _, _ = apply_selected_review(str(tmp_path), 'English', ['0'])
    assert applied == 1
    assert json.loads((tmp_path/'translation.json').read_text())[0]['translation'] == "You don't have to work tomorrow."


def test_source_validation_rejects_truncation_and_unsupported_speaker_change(tmp_path, monkeypatch):
    from tools.source_validation import validate_source_repairs
    source = [{'text': '你为什么迟到小李先坐下等我回来', 'speaker': 'A', 'start': 0, 'end': 4}]
    proposals = [{'index': 0, 'text': '小李', 'speaker': 'B', 'skip': True}]
    monkeypatch.setattr('tools.translation_backends.llm_translate', lambda *a, **k: json.dumps({
        'decisions': [{'proposal': 0, 'text_approved': False, 'speaker_approved': False,
                       'skip_approved': False, 'reason': '刪去完整問答且無法確認講者'}]}))
    assert validate_source_repairs(str(tmp_path), source, proposals, 'OpenAI', 'asr_repair') == [{'index': 0}]
    assert proposals[0]['text'] == '小李' and source[0]['speaker'] == 'A'


def test_source_validation_approvals_are_independent(tmp_path, monkeypatch):
    from tools.source_validation import validate_source_repairs
    source = [{'text': '今天放加', 'speaker': 'A', 'source_subtitle_evidence': ['今天放假']}]
    monkeypatch.setattr('tools.translation_backends.llm_translate', lambda *a, **k: json.dumps({
        'decisions': [{'proposal': 0, 'text_approved': True, 'speaker_approved': False,
                       'skip_approved': False, 'reason': '字幕支持改字，不支持換講者'}]}))
    result = validate_source_repairs(str(tmp_path), source,
        [{'index': 0, 'text': '今天放假', 'speaker': 'B'}], 'OpenAI', 'asr_repair')
    assert result == [{'index': 0, 'text': '今天放假'}]


@pytest.mark.parametrize('response', ['{}', '{"decisions":[]}', 'not JSON'])
def test_incomplete_source_validation_keeps_source_and_reports_failure(tmp_path, monkeypatch, response):
    from tools.source_validation import validate_source_repairs
    from tools.review_state import failures
    monkeypatch.setattr('tools.translation_backends.llm_translate', lambda *a, **k: response)
    source = [{'text': '你先回去，我留下', 'speaker': 'A'}]
    result = validate_source_repairs(str(tmp_path), source,
        [{'index': 0, 'text': '我先回去'}], 'OpenAI', 'asr_repair')
    assert result == [{'index': 0}]
    assert failures(str(tmp_path), 'English')['原文改寫覆核'] == 'Incomplete'


def test_asr_pipeline_keeps_full_original_when_rewrite_is_rejected(tmp_path, monkeypatch):
    from tools import asr_repair
    source = [{'text': '你为什么迟到？小李，你先坐下，我马上回来。',
               'speaker': 'A', 'start': 0, 'end': 5}]
    def respond(method, messages, **kwargs):
        if kwargs.get('purpose') == 'review':
            return json.dumps({'decisions': [{'proposal': 0, 'text_approved': False,
                'speaker_approved': False, 'skip_approved': False, 'reason': '刪去問答'}]})
        return json.dumps({'fixes': [{'index': 0, 'text': '小李', 'speaker': 'B'}]})
    monkeypatch.setattr('tools.translation_backends.llm_translate', respond)
    monkeypatch.setattr(asr_repair, '_speaker_audit', lambda rows, bible, method, allowed, created, *a, **k:
                        (0, created, rows))
    assert asr_repair._ai_repair(str(tmp_path), source, {}, 'OpenAI')[-1] == source


def test_missing_speaker_reference_uses_source_time_and_keeps_existing_clip(tmp_path):
    import numpy as np
    import soundfile as sf
    from tools.asr import ensure_speaker_audio
    rate = 24000
    sf.write(tmp_path/'audio_vocals.wav', np.concatenate([np.zeros(rate), np.ones(rate)*.25, np.zeros(rate)]), rate)
    (tmp_path/'SPEAKER').mkdir()
    sf.write(tmp_path/'SPEAKER/A.wav', np.ones(100)*.1, rate)
    original = (tmp_path/'SPEAKER/A.wav').read_bytes()
    rows = [{'speaker': 'A', 'start': 0, 'end': 1},
            {'speaker': 'B', 'start': 2, 'end': 3, 'orig_start': 1, 'orig_end': 2}]
    ensure_speaker_audio(str(tmp_path), rows)
    samples, _ = sf.read(tmp_path/'SPEAKER/B.wav')
    assert np.mean(np.abs(samples)) > .2
    assert (tmp_path/'SPEAKER/A.wav').read_bytes() == original


@pytest.mark.parametrize('text,expected', [('Hehehehe', True), ('ha ha!', True), ('哈哈哈', True),
    ('He left.', False), ('he has to go', False), ('哈哈，你先走', False)])
def test_laughter_detection_requires_a_complete_nonverbal_card(text, expected):
    from tools.source_vocalizations import is_laughter_card
    assert is_laughter_card(text) is expected


def test_original_laughter_keeps_source_duration_and_reuses_cache(tmp_path):
    import numpy as np
    import soundfile as sf
    from tools.source_vocalizations import preserve_laughter
    rate = 24000
    samples = np.sin(np.arange(rate)*2*np.pi*220/rate).astype(np.float32)*.2
    sf.write(tmp_path/'audio_vocals.wav', samples, rate)
    output = str(tmp_path/'laugh.wav')
    line = {'text': 'Hehehehe', 'start': .2, 'end': .6}
    assert preserve_laughter(str(tmp_path), line, output, samples)
    actual, _ = sf.read(output)
    assert len(actual) == round(.4*rate)
    np.testing.assert_allclose(actual, samples[4800:14400], atol=4e-5)
    stamp = (tmp_path/'laugh.wav').stat().st_mtime_ns
    assert preserve_laughter(str(tmp_path), line, output, samples)
    assert (tmp_path/'laugh.wav').stat().st_mtime_ns == stamp
    assert not preserve_laughter(str(tmp_path), {**line, 'text': 'He came back.'}, output, samples)


def test_generated_context_cannot_override_current_glossary(tmp_path):
    from tools.translation_bible import review_bible_context
    (tmp_path/'source_bible.json').write_text(json.dumps({'summary': '角色讨论新计划',
        'outline': '角色解释计划', 'glossary': '温风=Long Provisional Wind Name'}))
    context = review_bible_context(str(tmp_path), {'summary': 'They discuss a plan.',
        'glossary': '温风=Warm Breeze'})
    assert 'Long Provisional Wind Name' not in context
    assert 'Warm Breeze' in context and '角色解释计划' in context
    assert 'independent caption evidence' in context


def test_translation_cannot_add_new_asr_words_after_source_approval(tmp_path, monkeypatch):
    from tools.translation import _prepare_source_lines
    rows = [{'text': '大家明天正常上班', 'speaker': 'A', 'start': 0, 'end': 2}]
    monkeypatch.setattr('tools.asr_repair.already_repaired', lambda *a: True)
    monkeypatch.setattr('tools.asr_repair.ensure_repaired_transcript', lambda folder, lines: lines)
    def unexpected(*a, **k):
        pytest.fail('Approved source must not acquire unreviewed ASR words in translation')
    monkeypatch.setattr('tools.asr_gaps.recover_leading_speech', unexpected)
    assert _prepare_source_lines(rows, 'English', str(tmp_path)) == rows
