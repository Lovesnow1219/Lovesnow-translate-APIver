import json
from copy import deepcopy

from tools.source_subtitles import apply_evidence, pending, preserves_caption_terms, set_enabled


def test_caption_replacements_require_matching_frame_and_literal_evidence():
    source = [{'text': '明天有三个月假期', 'start': 0, 'end': 2, 'speaker': 'A'}]
    correction = {'index': 0, 'evidence': [{'frame': 2, 'text': '明天有三天假期'}],
                  'edits': [{'from': '三个月', 'to': '三天'}]}
    original = deepcopy(source)
    fixed, accepted = apply_evidence(source, [correction], {2: 0})
    assert fixed[0]['text'] == '明天有三天假期'
    assert fixed[0]['end'] == 2 and fixed[0]['speaker'] == 'A'
    assert source == original and len(accepted) == 1
    assert preserves_caption_terms(fixed[0], '明天有三天假期！')
    assert not preserves_caption_terms(fixed[0], '明天有三个月假期')
    assert apply_evidence(source, [correction], {2: 1}) == (source, [])
    invented = deepcopy(correction)
    invented['edits'][0]['to'] = '三年'
    assert apply_evidence(source, [invented], {2: 0}) == (source, [])


def test_uncertain_empty_or_destructive_caption_edits_leave_source_intact():
    source = [{'text': '你先走，我留下来', 'start': 0, 'end': 2}]
    corrections = [None, {'index': -1}, {'index': True},
        {'index': 0, 'evidence': [], 'edits': [{'from': '你先走', 'to': '我先走'}]},
        {'index': 0, 'evidence': [{'frame': 1, 'text': '你先走'}],
         'edits': [{'from': '我留下来', 'to': ''}]}]
    assert apply_evidence(source, corrections, {1: 0}) == (source, [])


def test_disabled_visual_review_never_requires_video_or_api(tmp_path, monkeypatch):
    from tools import source_subtitles as mod
    monkeypatch.setattr(mod, '_frame', lambda *a: (_ for _ in ()).throw(AssertionError('No frame request')))
    source = [{'text': '测试'}]
    assert mod.review_source_subtitles(str(tmp_path), source) == source
    set_enabled(str(tmp_path), False)
    assert not pending(str(tmp_path))
    assert json.loads((tmp_path/mod.SETTINGS).read_text())['enabled'] is False


def test_multimodal_chat_compatibility_preserves_image_and_original_input():
    from tools.translation_openai import _chat_messages
    messages = [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'Check this caption'},
        {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,test', 'detail': 'high'}]}]
    original = deepcopy(messages)
    converted = _chat_messages(messages)
    assert converted[0]['content'][1] == {'type': 'image_url', 'image_url': {
        'url': 'data:image/jpeg;base64,test', 'detail': 'high'}}
    assert messages == original


def test_caption_sampling_covers_long_cards_and_asr_boundary_error():
    from tools.source_subtitles import sample_times, MAX_FRAMES
    lines = [{'start': 40, 'end': 45}, {'start': 46.75, 'end': 47.4}]
    plan = sample_times(lines, [0, 1], 50)
    assert plan[0][0] == 39.7 and plan[0][-1] == 45.3
    assert max(b-a for a,b in zip(plan[0],plan[0][1:])) <= 1
    assert min(plan[1]) < 46.75 and max(plan[1]) > 47.4
    many = [{'start': i*10, 'end': i*10+9} for i in range(MAX_FRAMES)]
    plan = sample_times(many, list(range(MAX_FRAMES)), 500)
    assert len(plan) == MAX_FRAMES
    assert sum(map(len, plan.values())) == MAX_FRAMES


def test_whole_caption_override_requires_two_distinct_matching_frames():
    source = [{'text': '可惜了', 'start': 1, 'end': 2}]
    item = {'index': 0, 'evidence': [{'frame': 0, 'text': '没关系'}],
            'edits': [{'from': '可惜了', 'to': '没关系'}]}
    assert apply_evidence(source, [item], {0: 0}) == (source, [])
    item['evidence'].append({'frame': 0, 'text': '没关系'})
    assert apply_evidence(source, [item], {0: 0}) == (source, [])
    item['evidence'].append({'frame': 1, 'text': '没关系'})
    fixed, accepted = apply_evidence(source, [item], {0: 0, 1: 0})
    assert fixed[0]['text'] == '没关系' and len(accepted) == 1


def test_failed_review_remains_visible_and_cannot_be_cached_as_passed(tmp_path, monkeypatch):
    from tools import translation_review as review, translation_backends
    from tools.review_state import failures
    def fail(*a, **k):
        raise TimeoutError('provider connection timed out')
    monkeypatch.setattr(translation_backends, 'llm_translate', fail)
    rows = [{'text': '你好', 'translation': 'Hello.', 'start': 0, 'end': 1, 'speaker': 'A'}]
    review._after_dub_ai(str(tmp_path), rows, 'English', {}, [0], 'OpenAI')
    assert failures(str(tmp_path), 'English') == {'配音後審稿': 'TimeoutError'}
    assert '審核未完成' in review.review_status_line(str(tmp_path), 'English')
    review.stamp_after_dub(str(tmp_path), 'English')
    assert not review.after_dub_already_done(str(tmp_path), 'English')
    monkeypatch.setattr(translation_backends, 'llm_translate', lambda *a, **k: '{"findings":[]}')
    review._after_dub_ai(str(tmp_path), rows, 'English', {}, [0], 'OpenAI')
    assert failures(str(tmp_path), 'English') == {}


def test_episode_brief_reaches_bible_generation(monkeypatch):
    from tools import translation_bible as mod
    messages = []
    def respond(method, request):
        messages.extend(request)
        return json.dumps({'title': 'Example', 'summary': 'A workplace discussion.',
                           'outline': 'The characters discuss work.', 'glossary': '', 'voices': ''})
    monkeypatch.setattr(mod, 'llm_translate', respond)
    brief = 'Preserve ordinary workplace terminology and dry humor.'
    mod.build_dubbing_bible({'title': 'Example', 'uploader': 'Example', 'dubbing_style': brief},
        [{'text': '大家正在讨论新的工作安排和待遇。' * 8}], 'English')
    assert any(brief in row['content'] for row in messages)


def test_failed_caption_review_is_visible_for_all_target_languages(tmp_path):
    from tools.review_state import warning
    set_enabled(str(tmp_path), True)
    (tmp_path/'source_subtitle_review.json').write_text(json.dumps({
        'completed': False, 'errors': ['TimeoutError']}))
    assert '原片字幕對照' in warning(str(tmp_path), 'English')
    assert '原片字幕對照' in warning(str(tmp_path), 'Japanese')
    set_enabled(str(tmp_path), False)
    assert not warning(str(tmp_path), 'English')


def test_post_dub_semantic_finding_survives_without_timing_flags(monkeypatch):
    from tools import translation_review as review
    monkeypatch.setattr(review, '_after_dub_flags', lambda *a, **k: ([], 2.0, 1.0, 0.0))
    rows = [{'text': '明天放假', 'translation': 'Work tomorrow.', 'start': 0, 'end': 2}]
    item = {'index': 0, 'issue': 'The translation reverses the meaning.', 'suggest': 'Tomorrow is a day off.'}
    assert review._keep_after_dub_finding(None, rows, item, 'English')


def test_malformed_review_is_not_reported_as_zero_findings(tmp_path):
    from tools import translation_review as review
    from tools.review_state import begin_stage, failures
    for response in ('', 'not JSON', '{bad JSON}', '{}'):
        assert review._parse_review_batch(response, [0], str(tmp_path), 'English', '配音後審稿') == []
        assert failures(str(tmp_path), 'English')['配音後審稿'] == 'ValueError'
        begin_stage(str(tmp_path), 'English', '配音後審稿')
    assert review._parse_review_batch('{"findings":[]}', [0], str(tmp_path), 'English', '配音後審稿') == []
    assert not failures(str(tmp_path), 'English')
