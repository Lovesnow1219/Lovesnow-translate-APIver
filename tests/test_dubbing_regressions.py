import json
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.io import wavfile

from tools.dubbing_settings import (file_identity, save_delivery_settings, stamp_render,
                                    render_is_current, validate_tts_speed)
from tools.dubbing_timing import trim_edge_silence
from tools.tts_mix import assemble_dub_timeline, fit_tts_to_gap
from tools.synthesize import synthesize_video, _run_ffmpeg, generate_srt

SR = 24000


def tone(seconds, frequency=440):
    return (0.15 * np.sin(2 * np.pi * frequency * np.arange(round(SR * seconds)) / SR)).astype(np.float32)


def write_wav(path, samples):
    wavfile.write(path, SR, (samples * 32767).astype(np.int16))
    return str(path)


def line(start, end, text='这是真的吗', translation='Is this real?'):
    return dict(start=start, end=end, text=text, translation=translation, speaker='SPEAKER_00')


def test_edge_trim_keeps_internal_pause_and_quiet_breath():
    breath = tone(.1) * .006
    middle = np.concatenate([breath, tone(.4), np.zeros(SR // 2), tone(.4), breath])
    samples = np.concatenate([np.zeros(SR // 2), middle, np.zeros(SR // 2)])
    out = trim_edge_silence(samples, SR)
    assert len(out) == pytest.approx(len(middle) + SR * .2, abs=4)
    assert np.count_nonzero(out) == np.count_nonzero(samples)
    assert np.array_equal(trim_edge_silence(np.zeros(SR), SR), np.zeros(SR))


def test_full_sentence_survives_overflow_and_later_gap_recovers(tmp_path):
    paths = [write_wav(tmp_path / f'{i}.wav', tone(n)) for i, n in enumerate([2, .5, .5])]
    lines = [line(0, 1), line(1, 1.5), line(4, 4.5)]
    result = assemble_dub_timeline(lines, paths)
    assert lines[1]['start'] == pytest.approx(2)
    assert lines[2]['start'] == pytest.approx(4)
    assert lines[0]['orig_start'] == 0
    assert len(result) / SR == pytest.approx(4.5)
    assert lines[1]['dub_timing']['delay_seconds'] == pytest.approx(1)
    wav, duration = fit_tts_to_gap(paths[0], .25)
    assert duration == pytest.approx(2)


def test_long_edge_silence_does_not_delay_next_line(tmp_path):
    path = write_wav(tmp_path / 'padded.wav', np.concatenate([np.zeros(SR//2), tone(1), np.zeros(SR//2)]))
    next_path = write_wav(tmp_path / 'next.wav', tone(.5))
    lines = [line(0, 1.2), line(1.3, 1.8)]
    assemble_dub_timeline(lines, [path, next_path])
    assert lines[1]['start'] == pytest.approx(1.3)
    assert lines[0]['dub_timing']['trimmed_seconds'] > .79


def test_timeline_rejects_missing_audio_and_out_of_order(tmp_path):
    with pytest.raises(ValueError, match='數量'):
        assemble_dub_timeline([line(0, 1)], [])
    with pytest.raises(ValueError, match='順序'):
        assemble_dub_timeline([line(2, 3), line(0, 1)], ['a', 'b'])


def test_skipped_bgm_does_not_advance_timeline(tmp_path):
    from tools.line_roles import should_skip_dub
    skipped = line(0, 1)
    skipped['line_role'] = 'bgm'
    if not should_skip_dub(skipped):
        skipped['dub'] = False
    # Explicitly use the same classification entry point as production.
    with patch('tools.line_roles.should_skip_dub', side_effect=lambda item: item is skipped):
        lines = [skipped, line(.02, .52)]
        path = write_wav(tmp_path / 'speech.wav', tone(.5))
        assemble_dub_timeline(lines, ['missing-placeholder.wav', path])
    assert lines[1]['start'] == .02


@pytest.fixture
def movie(tmp_path):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
        'testsrc2=size=320x180:rate=30:duration=2', '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', str(tmp_path/'download.mp4')], check=True)
    write_wav(tmp_path/'audio_combined.wav', tone(3))
    (tmp_path/'translation.json').write_text(json.dumps([line(0, 1), line(1, 2)]))
    return tmp_path


def stream_durations(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
        'stream=codec_type,duration', '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    return {s['codec_type']: float(s['duration']) for s in json.loads(result.stdout)['streams']}


@pytest.mark.parametrize('speed', [.5, 1, 1.5, 2])
def test_mux_preserves_entire_audio_and_video_at_all_speeds(movie, speed):
    result = synthesize_video(str(movie), subtitles=False, speed_up=speed, resolution='144p')
    durations = stream_durations(result)
    assert durations['audio'] == pytest.approx(3/speed, abs=.08)
    assert durations['video'] == pytest.approx(3/speed, abs=.08)


def test_mux_cache_reuses_identical_settings_and_rebuilds_changes(movie):
    result = synthesize_video(str(movie), subtitles=True, resolution='144p')
    identity = file_identity(result)
    assert synthesize_video(str(movie), subtitles=True, resolution='144p') == result
    assert file_identity(result) == identity
    synthesize_video(str(movie), subtitles=True, resolution='144p', subtitle_position='top')
    assert file_identity(result) != identity
    identity = file_identity(result)
    synthesize_video(str(movie), subtitles=True, speed_up=2, resolution='144p')
    assert stream_durations(result)['video'] == pytest.approx(1.5, abs=.08)
    assert file_identity(result) != identity


def test_volume_works_without_extra_bgm(movie):
    result = synthesize_video(str(movie), subtitles=False, video_volume=0, resolution='144p')
    samples = subprocess.run(['ffmpeg', '-v', 'error', '-i', result, '-f', 'f32le', '-ac', '1', '-'],
                              check=True, capture_output=True).stdout
    assert np.max(np.abs(np.frombuffer(samples, dtype=np.float32))) < .0001


@pytest.mark.parametrize('position', ['top', 'bottom'])
def test_subtitles_render_in_the_selected_part_of_the_picture(movie, position):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
        'color=black:size=320x180:rate=30:duration=2', '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', str(movie/'download.mp4')], check=True)
    result = synthesize_video(str(movie), resolution='144p', subtitle_position=position)
    frame = subprocess.run(['ffmpeg', '-v', 'error', '-ss', '0.3', '-i', result,
        '-frames:v', '1', '-vf', 'scale=256:144', '-pix_fmt', 'gray', '-f', 'rawvideo', '-'],
        capture_output=True, check=True).stdout
    pixels = np.frombuffer(frame, np.uint8).reshape(144, 256)
    # Small anti-aliased glyphs need not contain fully white pixel interiors.
    ys, xs = np.where(pixels > 80)
    assert len(ys) > 10
    assert .35 < xs.mean()/256 < .65
    assert (ys.max() < 36) if position == 'top' else (ys.min() > 108)


def test_failed_encode_does_not_report_old_output_as_success(movie):
    result = synthesize_video(str(movie), subtitles=False, resolution='144p')
    before = Path(result).read_bytes()
    assert not _run_ffmpeg(['ffmpeg', '-y', '-i', str(movie/'does-not-exist'), result], duration=3)
    assert Path(result).read_bytes() == before


def test_subtitles_use_original_picture_time_and_playback_speed(tmp_path):
    cue = line(5, 7)
    cue.update(orig_start=1, orig_end=3)
    path = tmp_path/'sub.srt'
    generate_srt([cue], str(path), speed_up=2)
    assert '00:00:00,500 --> 00:00:01,500' in path.read_text()


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), 2])
def test_invalid_tts_speed_is_rejected(value):
    with pytest.raises(ValueError):
        validate_tts_speed(value)


def test_mix_cache_changes_with_text_voice_speed_and_wav(tmp_path):
    lines = [line(0, 1)]
    (tmp_path/'translation.json').write_text(json.dumps(lines))
    (tmp_path/'wavs').mkdir()
    write_wav(tmp_path/'wavs/0000.wav', tone(1))
    stamp_render(str(tmp_path), 'English')
    assert render_is_current(str(tmp_path), 'English')
    assert not render_is_current(str(tmp_path), 'Japanese')
    save_delivery_settings(str(tmp_path), 1.05)
    assert not render_is_current(str(tmp_path), 'English')
    stamp_render(str(tmp_path), 'English')
    lines[0]['translation'] = 'Really?'
    (tmp_path/'translation.json').write_text(json.dumps(lines))
    assert not render_is_current(str(tmp_path), 'English')
    stamp_render(str(tmp_path), 'English')
    (tmp_path/'speaker_voices.json').write_text('{"SPEAKER_00": {"id": "new"}}')
    assert not render_is_current(str(tmp_path), 'English')
    stamp_render(str(tmp_path), 'English')
    write_wav(tmp_path/'wavs/0000.wav', tone(.5))
    assert not render_is_current(str(tmp_path), 'English')


def test_parallel_translation_receives_neighbor_context(monkeypatch):
    from tools import translation
    lines = [line(0, 2, text='你是认真的吗'), line(2, 4, text='当然是假的')]
    observed = []
    monkeypatch.setattr('tools.api_keys.translate_workers_for_keys', lambda: (2, 1))
    def fake(*args, **kwargs):
        observed.append(kwargs['extra_note'])
        return args[1]['text'], ''
    monkeypatch.setattr(translation, '_translate_one', fake)
    assert translation._translate({}, lines, 'English', 'OpenAI') == [x['text'] for x in lines]
    assert all('你是认真的吗' in note and '当然是假的' in note for note in observed)
    assert all('ONLY' in note for note in observed)


def test_after_dub_review_invalidates_on_audio_change(tmp_path):
    from tools.translation_review import _after_dub_fingerprint
    lines = [line(0, 1)]
    (tmp_path/'wavs').mkdir()
    path = tmp_path/'wavs/0000.wav'
    write_wav(path, tone(1))
    before = _after_dub_fingerprint(lines, str(tmp_path))
    write_wav(path, tone(2))
    assert _after_dub_fingerprint(lines, str(tmp_path)) != before


def test_fish_request_cache_and_atomic_invalid_response(tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace
    from tools import tts_fish
    payload = io.BytesIO()
    wavfile.write(payload, SR, (tone(.3)*32767).astype(np.int16))
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs['json'])
        return SimpleNamespace(status_code=200, content=payload.getvalue(), text='')
    monkeypatch.setattr('tools.api_keys.next_api_key', lambda *a, **k: ('test-key', 1, 1))
    monkeypatch.setattr('tools.cost_tracker.record', lambda *a, **k: None)
    monkeypatch.setattr(tts_fish.requests, 'post', post)
    monkeypatch.setattr(tts_fish.time, 'sleep', lambda *a: None)
    path = str(tmp_path/'line.wav')
    tts_fish.tts('Really?', path, 'English', 'voice-a', speed=1)
    tts_fish.tts('Really?', path, 'English', 'voice-a', speed=1)
    assert len(calls) == 1
    tts_fish.tts('Really?', path, 'English', 'voice-b', speed=1)
    assert len(calls) == 2

    tts_fish.tts('Really?', path, 'English', 'voice-b', speed=1.05)
    assert len(calls) == 3
    assert calls[-1]['prosody']['speed'] == 1.05
    old = Path(path).read_bytes()
    monkeypatch.setattr(tts_fish.requests, 'post', lambda *a, **k:
        SimpleNamespace(status_code=200, content=b'{"error":"not audio"}', text=''))
    with pytest.raises(RuntimeError, match='WAV'):
        tts_fish.tts('New words', path, 'English', 'voice-b')
    assert Path(path).read_bytes() == old
    assert len(list(tmp_path.glob('*.wav'))) == 1


def test_fish_never_reuses_untracked_old_wav(tmp_path, monkeypatch):
    from tools import tts_fish
    path = write_wav(tmp_path/'line.wav', tone(.3))
    monkeypatch.setattr('tools.api_keys.next_api_key', lambda *a, **k: (_ for _ in ()).throw(ValueError('new request')))
    with pytest.raises(ValueError, match='new request'):
        tts_fish.tts('changed text', path, 'English')


def test_ambiguous_words_are_not_overwritten_by_another_genre():
    from tools.asr_repair import fix_realm_homophones
    from tools.translation_quality import sense_issue
    from tools.translation_prompts import _house_review_rules, dubbing_fixed_message
    assert fix_realm_homophones('普通河道需要清理') == '普通河道需要清理'
    assert sense_issue('你拿枪指着我？', "You're pointing a gun at me?") == ''
    assert '河道' not in _house_review_rules()
    assert 'A polearm 枪 is a spear' not in str(dubbing_fixed_message({}, 'English'))


def test_stopped_parallel_workers_finish_before_caller_returns():
    from tools.parallel import map_parallel
    done = threading.Event()
    started = threading.Event()
    def work(index):
        if index == 0:
            started.wait(1)
            raise RuntimeError('test failure')
        started.set()
        time.sleep(.05)
        done.set()
    with pytest.raises(RuntimeError):
        map_parallel(work, [0, 1], workers=2)
    assert done.is_set()


def test_style_notes_are_saved_per_episode(tmp_path):
    from tools.dubbing_settings import save_style_note, style_note
    from tools.translation_bible import bible_context
    summary = {'title': 'Test', 'summary': 'A comedy'}
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    save_style_note(str(tmp_path), 'Keep the deadpan reply.')
    saved = json.loads((tmp_path/'summary.json').read_text())
    assert 'Keep the deadpan reply.' in bible_context(saved)
    assert style_note(str(tmp_path)) == 'Keep the deadpan reply.'


def test_wav_review_measures_trimmed_audio(tmp_path):
    from tools.translation import _tts_wav_seconds
    (tmp_path/'wavs').mkdir()
    write_wav(tmp_path/'wavs/0000.wav', np.concatenate([np.zeros(SR), tone(1), np.zeros(SR)]))
    assert _tts_wav_seconds(str(tmp_path), 0) == pytest.approx(1.2, abs=.001)


def test_actual_tts_pipeline_uses_one_speed_for_short_and_long_lines(tmp_path, monkeypatch):
    from tools import tts
    from tools.translation_versions import write_translation
    lines = [line(0, 1, text='真的吗', translation='真的嗎？'),
             line(1, 2, text='这是一整句很长的话', translation='這是一整句很長的話，仍然要完整唸完。')]
    write_translation(str(tmp_path), lines, '繁體中文')
    write_wav(tmp_path/'audio_instruments.wav', np.zeros(SR*3))
    save_delivery_settings(str(tmp_path), 1.03)
    monkeypatch.setattr('tools.vocal_particles.recover_vocal_particles', lambda f, t, **k: t)
    monkeypatch.setattr('tools.vocal_particles.reclassify_particles_from_vocals', lambda f, t, *a: t)
    monkeypatch.setattr('tools.vocal_particles.repair_overlapping_particle_cards', lambda t, *a: t)
    monkeypatch.setattr('tools.target_language.speakers_are_locked', lambda *a: True)
    monkeypatch.setattr(tts, 'assign_speaker_voices', lambda *a, **k: ({}, False))
    monkeypatch.setattr(tts, 'voice_for_method', lambda *a, **k: 'voice-test')
    monkeypatch.setattr('tools.api_keys.workers_for_keys', lambda *a: (1, 1))
    def no_unreviewed_rewrite(*a, **k):
        pytest.fail('TTS must not overwrite approved dialogue with a word-budget rewrite')
    monkeypatch.setattr('tools.translation.tighten_overlong_lines', no_unreviewed_rewrite)
    speeds = []
    def fake_tts(text, path, **kwargs):
        speeds.append(kwargs['speed'])
        write_wav(path, tone(.5))
    monkeypatch.setattr(tts, 'fish_tts', fake_tts)
    combined, _ = tts.generate_wavs('Fish', str(tmp_path), '繁體中文')
    assert speeds == [1.03, 1.03]
    assert Path(combined).is_file()
    assert json.loads((tmp_path/'dub_meta.json').read_text())['translation'] == '繁体中文'
    assert render_is_current(str(tmp_path), '繁體中文')


def test_ui_contracts_and_local_http_configuration(monkeypatch):
    import inspect
    import webui
    from fastapi.testclient import TestClient
    from gradio.routes import App
    webui.app.queue()
    client = TestClient(App.create_app(webui.app))
    response = client.get('/config')
    assert response.status_code == 200
    config = response.json()
    labels = [c['props'].get('label', '') for c in config['components']]
    assert '配音語速（整片固定）' in labels
    assert '英文譯文字數預算（每秒詞）' in labels
    assert '本片翻譯風格／需求（可自由修改）' in labels
    assert '以原片字幕為翻譯依據' in labels
    assert '翻譯字幕位置' in labels
    for event in webui.app.fns.values():
        if event.fn in (webui.do_everything_with_cost, webui.synthesize_from_ui):
            inspect.signature(event.fn).bind(*([None] * len(event.inputs)))
        if event.fn is not webui.stop_running_job:
            assert event.concurrency_id == 'dubbing-project'
            assert event.concurrency_limit == 1
    observed = []
    monkeypatch.setattr(webui, 'generate_all_wavs_under_folder',
        lambda *a, **k: observed.append((a, k)))
    webui.tts_from_ui('test-folder', '繁體中文', 1.04)
    assert observed[0][0][2] == '繁体中文'
    assert observed[0][1]['tts_speed'] == 1.04


def test_empty_translation_placeholder_cannot_delay_speech(tmp_path):
    lines = [line(0, 1, translation=''), line(.02, .52)]
    silence = write_wav(tmp_path/'silent.wav', np.zeros(int(SR*.08)))
    speech = write_wav(tmp_path/'speech.wav', tone(.5))
    assemble_dub_timeline(lines, [silence, speech])
    assert lines[1]['start'] == .02
    assert lines[0]['dub_timing']['skipped']


def test_riff_repair_keeps_trailing_metadata_out_of_audio(tmp_path):
    from tools.tts_fish import _fix_wav_header
    path = tmp_path/'test.wav'
    write_wav(path, tone(.2))
    data = bytearray(path.read_bytes())
    data.extend(b'JUNK' + (4).to_bytes(4, 'little') + b'data')
    data[4:8] = (len(data)-8).to_bytes(4, 'little')
    path.write_bytes(data)
    _fix_wav_header(path)
    assert path.read_bytes() == data
    data = data[:-12]
    data[4:8] = b'\xff'*4
    data[40:44] = b'\xff'*4
    path.write_bytes(data)
    _fix_wav_header(path)
    import wave
    with wave.open(str(path)) as stream:
        assert stream.getnframes() == int(SR*.2)


def test_cancellation_does_not_poison_next_job():
    from tools.job_control import start_job, request_stop, finish_job, check_stop, JobStopped
    job = start_job()
    try:
        with pytest.raises(RuntimeError, match='上一個任務'):
            start_job()
        request_stop()
        with pytest.raises(JobStopped):
            check_stop()
    finally:
        finish_job(job)
    check_stop()
    next_job = start_job()
    finish_job(next_job)


def test_exhausted_keys_are_not_recycled(monkeypatch):
    from tools import api_keys
    monkeypatch.setattr(api_keys, '_DEAD', {})
    monkeypatch.setattr(api_keys, '_COUNTERS', {})
    monkeypatch.setenv('TEST_DUB_KEY', 'first-secret, second-secret')
    api_keys.mark_api_key_dead('first-secret')
    assert api_keys.next_api_key('TEST_DUB_KEY', expand=False)[0] == 'second-secret'
    api_keys.mark_api_key_dead('second-secret')
    with pytest.raises(RuntimeError, match='均已停用'):
        api_keys.next_api_key('TEST_DUB_KEY', expand=False)


def test_usage_accounts_include_empty_response_before_fallback(monkeypatch):
    from types import SimpleNamespace as NS
    from tools import translation_openai as mod
    calls = []
    monkeypatch.setenv('OPENAI_API_KEY', 'private-main-key')
    monkeypatch.setenv('OPENAI_USE_RESPONSES', '1')
    first = NS(output_text='', output=[], usage=NS(input_tokens=10, output_tokens=20))
    second = NS(choices=[NS(message=NS(content='Hello'))], usage=NS(prompt_tokens=5, completion_tokens=3))
    client = NS(responses=NS(create=lambda **kw: first),
                chat=NS(completions=NS(create=lambda **kw: second)))
    monkeypatch.setattr(mod, 'OpenAI', lambda **kw: client)
    monkeypatch.setattr('tools.cost_tracker.record', lambda *args, **kw: calls.append(kw))
    assert mod._call_openai('private-main-key', 1, 1, [], reasoning_effort='off') == 'Hello'
    assert [c['response'] for c in calls] == [first, second]
    assert all(c['credential'] == 'OPENAI_API_KEY' for c in calls)
    assert 'private-main-key' not in repr(calls)


def test_timeout_does_not_start_second_generation_protocol(monkeypatch):
    from types import SimpleNamespace as NS
    from tools import translation_openai as mod
    def fail(**kwargs):
        raise TimeoutError('response timed out after submission')
    chat = []
    monkeypatch.setenv('OPENAI_USE_RESPONSES', '1')
    client = NS(responses=NS(create=fail), chat=NS(completions=NS(create=lambda **kw: chat.append(kw))))
    monkeypatch.setattr(mod, 'OpenAI', lambda **kw: client)
    with pytest.raises(TimeoutError):
        mod._call_openai('test-key', 1, 1, [])
    assert chat == []


def test_account_usage_sums_input_output_without_double_counting_reasoning():
    from tools.cost_tracker import credential_token_lines
    lines = credential_token_lines([
        {'provider':'openai', 'credential':'OPENAI_API_KEY', 'input_tokens':100,
         'output_tokens':50, 'reasoning_tokens':40},
        {'provider':'openai', 'credential':'OPENAI_API_KEY', 'input_tokens':20, 'output_tokens':10},
        {'provider':'openai', 'credential':'OPENAI_TRANSLATE_API_KEY', 'input_tokens':7, 'output_tokens':3},
        {'provider':'fish', 'characters':500},
    ])
    assert '- OPENAI_API_KEY：180' in lines
    assert '- OPENAI_TRANSLATE_API_KEY：10' in lines


def test_replicate_respects_proxy_and_no_proxy(monkeypatch):
    from tools.demucs import _replicate_client
    monkeypatch.setenv('REPLICATE_API_TOKEN', 'test-token')
    monkeypatch.setenv('REPLICATE_BASE_URL', 'https://api.replicate.com')
    monkeypatch.setattr('urllib.request.getproxies', lambda: {'https':'http://proxy.test:8080'})
    monkeypatch.setattr('urllib.request.proxy_bypass', lambda host: False)
    monkeypatch.setattr('replicate.Client', lambda **kwargs: kwargs)
    assert _replicate_client()['proxy'] == 'http://proxy.test:8080'
    monkeypatch.setattr('urllib.request.proxy_bypass', lambda host: True)
    assert 'proxy' not in _replicate_client()


def test_voice_sample_download_does_not_forward_api_secret(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS
    from tools import fish_voice_match as mod
    requests = []
    def get(url, **kwargs):
        requests.append((url, kwargs))
        if len(requests) == 1:
            return NS(status_code=200, json=lambda: {'samples':[{'audio':'https://media.example.test/voice.wav'}]})
        return NS(status_code=200, content=b'a'*4096, headers={})
    monkeypatch.setattr(mod, '_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(mod, '_auth_headers', lambda: {'Authorization':'Bearer private-fish-key'})
    monkeypatch.setattr(mod.requests, 'get', get)
    monkeypatch.setattr(mod, '_load_mono', lambda path: (np.ones(20), SR))
    monkeypatch.setattr('tools.utils.save_wav_norm', lambda *args, **kwargs: None)
    assert mod._download_sample('voice-id')
    assert requests[0][1]['headers']['Authorization'] == 'Bearer private-fish-key'
    assert 'headers' not in requests[1][1]


def test_mfcc_voice_cache_is_reused_without_rebuilding(monkeypatch, tmp_path):
    from tools import fish_voice_match as mod
    expected = np.array([.2,.3,.4])
    np.save(tmp_path/'cached-voice.npy', expected)
    (tmp_path/'meta.json').write_text(json.dumps({'kind':'mfcc'}))
    monkeypatch.setattr(mod, '_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(mod, '_ge2e_encoder_model', lambda: None)
    def unexpected(*args, **kwargs):
        pytest.fail('A valid MFCC cache should not download or synthesize audio')
    monkeypatch.setattr(mod, '_download_sample', unexpected)
    result=mod.ensure_stock_prints([{'id':'cached-voice'}], tts_fn=unexpected)
    np.testing.assert_array_equal(result['cached-voice'], expected)


def test_shortening_keeps_source_and_stops_on_no_safe_rewrite(monkeypatch):
    from tools import translation as mod
    source='我答应过要帮忙，但是我不能继续留下来。'
    current="I promised I'd help, but I can't stay."
    messages=[]
    def respond(method, incoming):
        messages.append(incoming)
        return current
    monkeypatch.setattr(mod, '_llm_translate', respond)
    result, prompt=mod._shorten_one(line(0, 1, source, current), 'English', 'OpenAI', [])
    assert result == current
    assert len(messages) == 1
    assert source in messages[0][-1]['content']


def test_dialogue_quote_punctuation_survives_cleanup():
    from tools.translation_quality import translation_postprocess
    text='You wrote “the center of the universe.”'
    assert translation_postprocess(text, 'English') == text
    assert translation_postprocess('“Wait, I am not done.”', 'English') == 'Wait, I am not done.'
    assert translation_postprocess("'Tis a strange day.", 'English') == "'Tis a strange day."


def test_usage_receipts_survive_without_finishing_session(tmp_path):
    from tools.cost_tracker import CostSession
    session=CostSession(str(tmp_path))
    try:
        session.add('openai','translate','gpt-5.6-luna',credential='OPENAI_TRANSLATE_API_KEY',input_tokens=100,output_tokens=40)
        session.add('openai','review','gpt-5.6-sol',credential='OPENAI_API_KEY',input_tokens=80,output_tokens=30)
        receipts=[json.loads(row) for row in (tmp_path/'api_usage.jsonl').read_text().splitlines()]
        assert [r['input_tokens']+r['output_tokens'] for r in receipts] == [140,110]
        assert all(r['run_id']==session.run_id for r in receipts)
        assert not (tmp_path/'cost.json').exists()
    finally:
        session.stop_ticker()


def test_sentence_initial_contraction_is_not_a_protected_character_name():
    from tools.translation_quality import suggest_wrecks_voice
    assert not suggest_wrecks_voice('你不是说三分钟送到吗？',
        "Didn't you promise delivery in three minutes?", 'You promised three minutes?', 'English')


def test_repeated_invalid_shortening_stops_without_five_paid_attempts(monkeypatch):
    from tools import translation as mod
    calls=[]
    current='I need you to stay and help me finish this.'
    def respond(*args, **kwargs):
        calls.append(1)
        return 'Go!'
    monkeypatch.setattr(mod, '_llm_translate', respond)
    monkeypatch.setattr(mod.time, 'sleep', lambda *a: None)
    result,_=mod._shorten_one(line(0,1,'我需要你留下来帮我做完这件事。',current),'English','OpenAI',[])
    assert result == current
    assert len(calls) == 2

@pytest.mark.parametrize('gap,same_speaker,merged', [(2.0, True, False), (.1, False, False), (.1, True, True)])
def test_asr_merge_text_and_timing_are_applied_together(monkeypatch, tmp_path, gap, same_speaker, merged):
    from tools import asr_repair as mod
    from tools import translation_backends
    cards = [line(0, 1, '今天下雨了'), line(1 + gap, 2 + gap, '记得带伞')]
    if not same_speaker:
        cards[1]['speaker'] = 'SPEAKER_01'
    original = json.loads(json.dumps(cards))
    monkeypatch.setattr(translation_backends, 'llm_translate', lambda *a, **k:
        json.dumps({'fixes': [{'index': 0, 'merge': [1], 'text': '今天下雨了，记得带伞'}]}))
    monkeypatch.setattr('tools.source_validation.validate_source_repairs', lambda folder, rows, fixes, *a: fixes)
    monkeypatch.setattr(mod, '_speaker_audit', lambda rows, bible, method, allowed, created, progress, **k:
        (0, created, rows))
    result = mod._ai_repair(str(tmp_path), cards, {}, 'OpenAI')[-1]
    if merged:
        assert len(result) == 1
        assert result[0]['text'] == '今天下雨了，记得带伞'
        assert result[0]['orig_end'] == 2 + gap
    else:
        assert result == original


@pytest.mark.parametrize('source,translation,language', [
    ('嘿嘿嘿', 'Heh-heh-heh.', 'English'),
    ('呵呵', 'Heh.', 'English'),
    ('嗯', 'うん。', 'Japanese'),
])
def test_standalone_interjections_do_not_gain_a_second_lead(source, translation, language):
    from tools.vocal_particles import ensure_particle_translation_lead, particle_lead_in
    row = line(0, 1, source, translation)
    assert particle_lead_in(source, language) == ''
    for _ in range(3):
        ensure_particle_translation_lead(row, language)
    assert row['translation'] == translation


def test_interjection_lead_still_applies_once_to_a_full_sentence():
    from tools.vocal_particles import ensure_particle_translation_lead, particle_lead_in
    row = line(0, 2, '嗯，你先走吧。', 'You go ahead.')
    lead = particle_lead_in(row['text'], 'English')
    assert lead
    ensure_particle_translation_lead(row, 'English')
    ensure_particle_translation_lead(row, 'English')
    assert row['translation'] == lead + 'You go ahead.'
