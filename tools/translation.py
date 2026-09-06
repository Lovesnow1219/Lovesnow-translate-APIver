# -*- coding: utf-8 -*-
import json
import os
import time

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from tools.asr_source import (
    split_long_segments,
    _is_particle_only,
)
from tools.translation_machine import translator_response
from tools.target_language import (
    drop_asr_junk_lines,
    is_asr_junk,
    is_chinese_target,
    needs_translation_refresh,
    purge_asr_junk_folder,
    required_translation_version,
    translation_language,
    uses_char_budget,
    uses_word_budget,
)
from tools.translation_backends import emergency_translate as _emergency_translate
from tools.translation_backends import llm_translate as _llm_translate
from tools.translation_bible import (
    OUTLINE_VERSION,
    _load_folder_info,
    _source_transcript,
    ensure_episode_bible,
)
from tools.translation_prompts import dubbing_fixed_message as _dubbing_fixed_message
from tools.translation_prompts import review_rewrite_rules
from tools.translation_quality import (
    _extra_budget_scale,
    _folder_has_leftover_chinese,
    _forbids_chinese_script,
    _slot_seconds,
    _char_slack,
    _spoken_char_budget,
    _spoken_word_budget,
    _translate_user_content,
    _word_slack,
    leftover_chinese_indices,
    should_not_tighten,
    split_sentences,
    _sense_broken,
    suggest_wrecks_voice,
    translation_postprocess,
    usable_target_text as _usable_target_text,
    valid_translation,
    _foreign_asr_source,
    _two_mouth_source,
)

def _prepare_source_lines(transcript, target_language, folder=None):
    from tools.vocal_particles import recover_vocal_particles, repair_overlapping_particle_cards

    lines = list(transcript or [])
    repaired = False
    if folder:
        from tools.asr_repair import already_repaired, ensure_repaired_transcript
        repaired = already_repaired(folder)
        if not repaired:
            lines = recover_vocal_particles(folder, lines)
            from tools.asr_gaps import recover_missing_speech
            lines = recover_missing_speech(folder, lines)
        else:
            from tools.asr_gaps import recover_leading_speech
            from tools.line_roles import promote_bgm_speech
            promote_bgm_speech(lines)
            lines = recover_leading_speech(folder, lines)
        lines = ensure_repaired_transcript(folder, lines)
        repaired = already_repaired(folder)
    if not repaired and (uses_word_budget(target_language) or uses_char_budget(target_language)):
        lines = split_long_segments(lines)
    lines, junk_n = drop_asr_junk_lines(lines)
    if junk_n:
        logger.info(f'翻譯前丟掉 {junk_n} 條辨識幻聽')
    return repair_overlapping_particle_cards(lines, target_language)



def repair_shared_orig_windows(transcript):
    """Give split cards their own time slots so a leading 嗯/Hmm is not crushed to 0.08s."""
    i = 0
    n = len(transcript or [])
    while i < n:
        origin = transcript[i].get('orig_start')
        finish = transcript[i].get('orig_end')
        j = i + 1
        while (
            j < n
            and origin is not None
            and transcript[j].get('orig_start') == origin
            and transcript[j].get('orig_end') == finish
        ):
            j += 1
        if j > i + 1 and origin is not None and finish is not None:
            span = max(0.05, float(finish) - float(origin))
            weights = [
                max(1, len((line.get('translation') or line.get('text') or '').strip()))
                for line in transcript[i:j]
            ]
            total = sum(weights) or 1
            cursor = float(origin)
            for offset, line in enumerate(transcript[i:j]):
                piece = span * (weights[offset] / total)
                line['orig_start'] = round(cursor, 3)
                line['orig_end'] = round(float(finish) if offset == j - i - 1 else cursor + piece, 3)
                cursor = line['orig_end']
        i = j
    return transcript


def _translate_one(summary, line, target_language, method, fixed_message, history=None, budget_scale=1.0, extra_note=''):
    text = line['text']
    if is_asr_junk(text):
        logger.info(f'跳過辨識噪音：{(text or "")[:24]}')
        return '', ''
    if _is_particle_only(text):
        from tools.vocal_particles import particle_translation
        ready = particle_translation(text, target_language)
        if ready:
            return ready, ''
    from tools.line_roles import should_skip_dub
    if should_skip_dub(line):
        logger.info(f'跳過不配（配樂或殘句）：{(text or "")[:24]}')
        return '', ''
    if translation_language(target_language) == 'English' and _foreign_asr_source(text):
        logger.info(f'外文辨識照抄：{(text or "")[:40]}')
        return (text or '').strip(), ''
    from tools.vocal_particles import (
        collapse_double_particle_lead,
        particle_lead_in,
        translation_has_particle_lead,
        _source_lead_glyph,
    )
    _particle_lead = particle_lead_in(text, target_language)
    _lead_glyph = _source_lead_glyph(text)

    def _finish(translation, user_content=''):
        translation = collapse_double_particle_lead(translation, target_language)
        if _particle_lead and translation:
            if not translation_has_particle_lead(translation, _lead_glyph, target_language):
                translation = _particle_lead + translation.lstrip()
            translation = collapse_double_particle_lead(translation, target_language)
        return translation, user_content

    duration = _slot_seconds(line)
    scale = float(budget_scale or 1.0) * _extra_budget_scale(line)
    extra = extra_note or ''
    if _two_mouth_source(text):
        extra = (
            'This card glued two speakers. Translate only the first speaker '
            '(words before the last 。！). Do not speak the other mouth\'s command. '
            + extra
        )
    if scale > 1.05 and 'system panel' not in extra:
        extra = (
            'This is a system panel, narration, or inner thought. A bit longer is OK. '
            + extra
        )
    user_content = _translate_user_content(
        text, duration, target_language, budget_scale=scale, extra_note=extra,
    )

    retry_message = 'Only translate the quoted sentence and give me the final translation.'
    translation = None
    last_model_output = ''
    if method == 'Google Translate':
        translation = translator_response(text, to_language=target_language, translator_server='google')
        return _finish(_usable_target_text(translation, target_language) or translation, user_content)
    if method == 'Bing Translate':
        translation = translator_response(text, to_language=target_language, translator_server='bing')
        return _finish(_usable_target_text(translation, target_language) or translation, user_content)

    history = history or []
    last_usable = ''
    for retry in range(5):
        messages = fixed_message + history[-30:] + [{'role': 'user', 'content': user_content}]
        if retry and retry_message:
            messages.append({'role': 'user', 'content': retry_message})
        try:
            response = _llm_translate(method, messages)
            last_model_output = (response or '').replace('\n', ' ').strip()
            logger.info(f'原文：{text}')
            logger.info(f'译文：{last_model_output}')
            usable = _usable_target_text(last_model_output, target_language)
            if usable:
                last_usable = usable
            success, cleaned = valid_translation(
                text, last_model_output, target_language, duration=duration, budget_scale=scale,
            )
            if not success:
                logger.warning(f'翻譯不合格：{cleaned}')
                retry_message = cleaned
                raise Exception('Invalid translation')
            return _finish(cleaned, user_content)
        except Exception as e:
            logger.error(e)
            logger.warning('翻译失败')
            if str(e) != 'Invalid translation':
                time.sleep(1)
    if last_usable:
        success, cleaned = valid_translation(
            text, last_usable, target_language, duration=duration, budget_scale=scale,
        )
        if success:
            logger.warning(f'翻譯多次未過檢查，改用合格後備: {cleaned[:80]}')
            return _finish(cleaned, user_content)
        logger.warning('翻譯多次未過檢查，不寫入不合格後備')
    if is_chinese_target(target_language):
        fallback = translation_postprocess(last_model_output, target_language)
        translation = fallback or text
        logger.warning(f'翻译多次失败，改用原文或模型原文: {translation[:80]}')
        return _finish(translation, user_content)
    emergency = _emergency_translate(text, target_language, method, fixed_message)
    if emergency:
        success, cleaned = valid_translation(
            text, emergency, target_language, duration=duration, budget_scale=scale,
        )
        if success:
            return _finish(cleaned, user_content)
        logger.warning('緊急翻譯也不合格，留空')
    logger.error('翻譯多次失敗，不把中文原文當譯文')
    return '', user_content


def _shorten_one(line, target_language, method, fixed_message, budget_scale=0.85):
    current = (line.get('translation') or '').strip()
    source = (line.get('text') or '').strip()
    if _foreign_asr_source(source):
        return source or current, ''
    duration = _slot_seconds(line)
    lang = translation_language(target_language)
    if uses_word_budget(target_language):
        budget = max(2, int(round(_spoken_word_budget(duration, target_language) * budget_scale)))
        unit = 'syllables' if lang == 'Vietnamese' else 'words'
        if lang == 'English':
            extra = (
                ' Cut filler only. Keep the speech-act, addressee, and names. '
                + review_rewrite_rules(target_language) +
                ' If the Chinese trails off, keep the named thing and the ellipsis. '
                ' Do not add somehow. Do not speak a second mouth. '
                ' Keep the verb in a question. A spear is not a gun. '
                ' Broken English is worse than being slightly long. Speak numbers.'
            )
        elif lang == 'Vietnamese':
            extra = ' Keep every clause. Speak numbers. Do not telegraph.'
        else:
            extra = ''
        user_content = (
            f'Shorten this dubbed line so it can be spoken in {duration:.1f}s, max {budget} {lang} {unit}. '
            f'Keep names and meaning.{extra} Output only the shortened line:"{current}"'
        )
    elif uses_char_budget(target_language):
        budget = max(4, int(round(_spoken_char_budget(duration, target_language) * budget_scale)))
        extra = (
            ' Keep は/が/を/の. Write invented names in katakana. Speak numbers. Do not telegraph.'
            if lang == 'Japanese' else ''
        )
        user_content = (
            f'Shorten this dubbed line so it can be spoken in {duration:.1f}s, max {budget} {lang} characters. '
            f'Keep names and meaning.{extra} Output only the shortened line:"{current}"'
        )
    else:
        return current, ''
    retry_message = 'Only output the shortened line.'
    last_model_output = ''
    for retry in range(5):
        messages = fixed_message + [{'role': 'user', 'content': user_content}]
        if retry and retry_message:
            messages.append({'role': 'user', 'content': retry_message})
        try:
            if method not in ('OpenAI', 'LLM', '阿里云-通义千问', 'Ollama', 'Ernie'):
                return current, user_content
            response = _llm_translate(method, messages)
            last_model_output = (response or '').replace('\n', ' ').strip()
            logger.info(f'原譯：{current}')
            logger.info(f'收緊：{last_model_output}')
            success, cleaned = valid_translation(
                source, last_model_output, target_language, duration=duration,
            )
            if not success:
                retry_message = cleaned
                raise Exception('Invalid translation')
            if uses_word_budget(target_language) and len(cleaned.split()) > budget:
                unit = 'syllables' if lang == 'Vietnamese' else 'spoken words'
                retry_message = f'Max {budget} {unit}. Output only the shortened line.'
                raise Exception('Still too long')
            if uses_char_budget(target_language) and len(cleaned) > budget:
                retry_message = f'Max {budget} characters. Output only the shortened line.'
                raise Exception('Still too long')
            if suggest_wrecks_voice(source, current, cleaned, target_language):
                retry_message = (
                    'That cut the person out. Keep the speech-act, names, and a real sentence. '
                    'Do not output a one-word bark. Output only the shortened line.'
                )
                raise Exception('Wrecked voice')
            return cleaned, user_content
        except Exception as exc:
            logger.warning(f'收緊失敗：{exc}')
            time.sleep(1)
    logger.warning('收緊放棄：沿用原譯')
    return current, user_content


def _tts_wav_seconds(folder, index):
    path = os.path.join(folder, 'wavs', f'{str(index).zfill(4)}.wav')
    if not os.path.isfile(path):
        return None
    import wave
    try:
        size = os.path.getsize(path)
        with wave.open(path, 'rb') as handle:
            rate = handle.getframerate() or 24000
            width = handle.getsampwidth() or 2
            channels = max(1, handle.getnchannels())
            frames = handle.getnframes()
            payload = max(0, size - 44)
            if frames > 10_000_000 or frames * width * channels > payload + 1024:
                return payload / float(rate * width * channels)
            return frames / float(rate)
    except Exception:
        return None


def _save_transcript(folder, transcript, target_language):
    from tools.translation_versions import write_translation

    write_translation(
        folder,
        transcript,
        target_language,
        translation_version=required_translation_version(target_language),
    )


def _use_language(folder, target_language):
    from tools.translation_versions import activate_language, has_version, snapshot_active

    snapshot_active(folder)
    if has_version(folder, target_language):
        activate_language(folder, target_language)


def _line_needs_tighten(folder, index, line, target_language, slack=None, ignore_wav_stale=False):
    from tools.vocal_particles import is_particle_card

    if is_particle_card(line.get('text')) or should_not_tighten(line.get('text')):
        return False, 1.0
    if _sense_broken(line.get('text'), line.get('translation')):
        return False, 1.0
    translation = line.get('translation') or ''
    duration = _slot_seconds(line)
    scale = _extra_budget_scale(line)
    if slack is None and uses_word_budget(target_language) and duration < 2.0:
        word_slack = 2
    else:
        word_slack = _word_slack(target_language) if slack is None else slack
    char_slack = _char_slack(target_language) if slack is None else slack
    if uses_word_budget(target_language) and len(translation.split()) > _spoken_word_budget(duration, target_language) * scale + word_slack:
        return True, 0.85
    if uses_char_budget(target_language) and len(translation) > _spoken_char_budget(duration, target_language) * scale + char_slack:
        return True, 0.85
    wav_sec = _tts_wav_seconds(folder, index)
    if wav_sec is not None and not ignore_wav_stale and _wav_is_stale(folder, index):
        wav_sec = None
    if wav_sec is not None and wav_sec > duration * 1.12 and wav_sec > duration + 0.25:
        spoken_n = len((translation or '').split())
        if spoken_n <= 2 and duration < 1.0:
            return False, 1.0
        return True, 0.8
    return False, 1.0


def _wav_is_stale(folder, index):
    wav_path = os.path.join(folder, 'wavs', f'{str(index).zfill(4)}.wav')
    trans_path = os.path.join(folder, 'translation.json')
    try:
        return os.path.getmtime(wav_path) + 0.05 < os.path.getmtime(trans_path)
    except OSError:
        return True


def tighten_overlong_lines(folder, target_language, method='OpenAI', indices=None, slack=None, ignore_wav_stale=False):
    _use_language(folder, target_language)
    path = os.path.join(folder, 'translation.json')
    with open(path, 'r', encoding='utf-8') as handle:
        transcript = json.load(handle)
    summary_path = os.path.join(folder, 'summary.json')
    if os.path.isfile(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as handle:
            summary = json.load(handle)
    else:
        summary = {'title': os.path.basename(folder), 'summary': ''}
    fixed_message = _dubbing_fixed_message(summary, target_language)
    changed = []
    wanted = None if indices is None else {int(i) for i in indices}
    for i, line in enumerate(transcript):
        if wanted is not None and i not in wanted:
            continue
        need, scale = _line_needs_tighten(
            folder, i, line, target_language, slack=slack, ignore_wav_stale=ignore_wav_stale,
        )
        if not need:
            continue
        new_text, _user = _shorten_one(
            line, target_language, method, fixed_message, budget_scale=scale,
        )
        old_text = (line.get('translation') or '').strip()
        if not new_text or new_text.strip() == old_text:
            continue
        line['translation'] = new_text
        from tools.vocal_particles import ensure_particle_translation_lead
        ensure_particle_translation_lead(line, target_language)
        changed.append(i)
        _clear_line_wavs(folder, i)
    if changed:
        _save_transcript(folder, transcript, target_language)
        _clear_mix_outputs(folder)
    logger.info(f'已收緊 {len(changed)} 句譯文：{folder}')
    return summary, transcript


def _clear_line_wavs(folder, index):
    wav_path = os.path.join(folder, 'wavs', f'{str(index).zfill(4)}.wav')
    for leftover in (wav_path, wav_path.replace('.wav', '.mp3')):
        if os.path.isfile(leftover):
            os.remove(leftover)


def _clear_mix_outputs(folder, keep_video=True):
    for name in ('audio_combined.wav', 'audio_tts.wav'):
        leftover = os.path.join(folder, name)
        if os.path.isfile(leftover):
            os.remove(leftover)
    if not keep_video:
        from tools.target_language import published_video_paths
        for leftover in published_video_paths(folder):
            if os.path.isfile(leftover):
                os.remove(leftover)


def _apply_speakers(old_lines, new_lines):
    if not old_lines:
        return new_lines
    for new in new_lines:
        start = float(new.get('start') or 0)
        end = float(new.get('end') or start)
        best = None
        best_overlap = 0.0
        for old in old_lines:
            old_start = float(old.get('orig_start') or old.get('start') or 0)
            old_end = float(old.get('orig_end') or old.get('end') or old_start)
            overlap = min(end, old_end) - max(start, old_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = old
        if best and best_overlap > 0.05 and best.get('speaker'):
            new['speaker'] = best['speaker']
    return new_lines


def refresh_translation_lines(folder, target_language, method='OpenAI'):
    """Re-translate from source so incomplete lines get full spoken sentences."""
    from tools.target_language import clear_tts_cache

    _use_language(folder, target_language)
    path = os.path.join(folder, 'translation.json')
    old_lines = []
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as handle:
            old_lines = json.load(handle)
    transcript_path = os.path.join(folder, 'transcript.json')
    if os.path.isfile(transcript_path):
        with open(transcript_path, 'r', encoding='utf-8') as handle:
            transcript = json.load(handle)
        transcript = _prepare_source_lines(transcript, target_language, folder=folder)
        _apply_speakers(old_lines, transcript)
    else:
        transcript = old_lines
    summary_path = os.path.join(folder, 'summary.json')
    if os.path.isfile(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as handle:
            summary = json.load(handle)
    else:
        summary = {'title': os.path.basename(folder), 'summary': ''}
    translations = _translate(summary, transcript, target_language, method)
    for i, line in enumerate(transcript):
        raw = translations[i] if i < len(translations) else ''
        if _forbids_chinese_script(target_language):
            raw = _usable_target_text(raw, target_language)
        line['translation'] = raw
    transcript = split_sentences(transcript, target_language=target_language)
    _save_transcript(folder, transcript, target_language)
    clear_tts_cache(folder)
    logger.info(f'已依大綱從原文重翻 {len(transcript)} 句：{folder}')
    _review_after_write(folder, target_language, method)
    return summary, transcript


def fill_changed_source_translations(folder, target_language, method='OpenAI'):
    """After gap-fill / source cleanup, translate only new or changed cards."""
    from tools.target_language import clear_tts_cache
    from tools.vocal_particles import card_end, card_start, ensure_particle_translation_lead

    _use_language(folder, target_language)
    src_path = os.path.join(folder, 'transcript.json')
    trans_path = os.path.join(folder, 'translation.json')
    if not os.path.isfile(src_path) or not os.path.isfile(trans_path):
        return 0
    with open(src_path, 'r', encoding='utf-8') as handle:
        source = json.load(handle)
    source = _prepare_source_lines(source, target_language, folder=folder)
    with open(src_path, 'w', encoding='utf-8') as handle:
        json.dump(source, handle, indent=4, ensure_ascii=False)
    with open(trans_path, 'r', encoding='utf-8') as handle:
        old_lines = json.load(handle)

    def _best_old(line):
        start, end = card_start(line), card_end(line)
        text = (line.get('text') or '').strip()
        best = None
        best_score = 0.0
        for old in old_lines:
            old_start = card_start(old)
            old_end = card_end(old)
            overlap = min(end, old_end) - max(start, old_start)
            if overlap <= 0.05:
                continue
            same = 1.0 if (old.get('text') or '').strip() == text else 0.0
            score = overlap + same * 10
            if score > best_score:
                best_score = score
                best = old
        return best

    summary_path = os.path.join(folder, 'summary.json')
    if os.path.isfile(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as handle:
            summary = json.load(handle)
    else:
        summary = {'title': os.path.basename(folder), 'summary': ''}
    fixed_message = _dubbing_fixed_message(summary, target_language)
    out = []
    changed = 0
    for line in source:
        row = dict(line)
        old = _best_old(line)
        src = (row.get('text') or '').strip()
        if old and (old.get('text') or '').strip() == src and (old.get('translation') or '').strip():
            row['translation'] = old.get('translation')
            if old.get('parent_text'):
                row['parent_text'] = old.get('parent_text')
            out.append(row)
            continue
        if _is_particle_only(src):
            from tools.vocal_particles import particle_translation
            row['translation'] = particle_translation(src, target_language)
        else:
            raw, _user = _translate_one(
                summary, row, target_language, method, fixed_message, history=[]
            )
            if _forbids_chinese_script(target_language):
                raw = _usable_target_text(raw, target_language)
            row['translation'] = raw
        ensure_particle_translation_lead(row, target_language)
        out.append(row)
        changed += 1
    if not changed:
        return 0
    _save_transcript(folder, out, target_language)
    clear_tts_cache(folder)
    logger.info(f'自動補翻 {changed} 句漏掉或改過的對白：{folder}')
    _review_after_write(folder, target_language, method)
    return changed


def refresh_leftover_chinese_lines(folder, target_language, method='OpenAI'):
    """Re-translate only lines that still contain Chinese in a non-Chinese target."""
    _use_language(folder, target_language)
    path = os.path.join(folder, 'translation.json')
    with open(path, 'r', encoding='utf-8') as handle:
        transcript = json.load(handle)
    dirty = leftover_chinese_indices(transcript, target_language)
    if not dirty:
        return True
    summary_path = os.path.join(folder, 'summary.json')
    if os.path.isfile(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as handle:
            summary = json.load(handle)
    else:
        summary = {'title': os.path.basename(folder), 'summary': ''}
    fixed_message = _dubbing_fixed_message(summary, target_language)
    changed = 0
    for i in dirty:
        new_text, _user = _translate_one(
            summary, transcript[i], target_language, method, fixed_message, history=[]
        )
        usable = _usable_target_text(new_text, target_language)
        if not usable:
            logger.error(f'第 {i} 句仍含中文，略過：{transcript[i].get("text")}')
            continue
        transcript[i]['translation'] = usable
        from tools.vocal_particles import ensure_particle_translation_lead
        ensure_particle_translation_lead(transcript[i], target_language)
        _clear_line_wavs(folder, i)
        changed += 1
    _save_transcript(folder, transcript, target_language)
    if changed:
        _clear_mix_outputs(folder)
    logger.info(f'已清掉 {changed}/{len(dirty)} 句殘留中文：{folder}')
    if changed:
        _review_after_write(folder, target_language, method)
    return summary, transcript



def _translate(summary, transcript, target_language='简体中文', method='LLM'):

    fixed_message = _dubbing_fixed_message(summary, target_language)

    from tools.api_keys import translate_workers_for_keys
    from tools.parallel import map_parallel
    if method in ('OpenAI', '阿里云-通义千问'):
        if method == 'OpenAI':
            workers, n_keys = translate_workers_for_keys()
        else:
            from tools.parallel import env_workers
            workers, n_keys = env_workers('TRANSLATE_WORKERS', 4), 1
    else:
        workers, n_keys = 1, 1
    if workers > 1 and len(transcript) > 1:
        extra = f'、{n_keys} 把 API key' if n_keys > 1 else ''
        logger.info(f'{method} 翻譯併發 {workers}{extra}')

        def _job(line):
            translation, _user = _translate_one(summary, line, target_language, method, fixed_message, history=[])
            return translation

        return map_parallel(_job, transcript, workers)

    history = []
    full_translation = []
    for line in transcript:
        translation, user_content = _translate_one(
            summary, line, target_language, method, fixed_message, history=history
        )
        full_translation.append(translation)
        history.append({'role': 'user', 'content': user_content})
        assistant_prefix = '' if not is_chinese_target(target_language) else '翻译：“'
        assistant_suffix = '' if not is_chinese_target(target_language) else '”'
        history.append({'role': 'assistant', 'content': f'{assistant_prefix}{translation}{assistant_suffix}'})
        time.sleep(0.1)
    return full_translation

def translate(method, folder, target_language='简体中文'):
    from tools.target_language import (
        _same_lang,
        load_dub_meta,
        translation_cache_ok,
    )
    junk_n = purge_asr_junk_folder(folder)
    if junk_n:
        logger.info(f'已從舊稿清掉 {junk_n} 條辨識幻聽：{folder}')
    _use_language(folder, target_language)
    if translation_cache_ok(folder, target_language):
        if _folder_has_leftover_chinese(folder, target_language):
            logger.info(f'譯文殘留中文，只重翻那些句：{folder}')
            return refresh_leftover_chinese_lines(folder, target_language, method)
        filled = fill_changed_source_translations(folder, target_language, method)
        if filled:
            return True
        logger.info(f'Translation already exists in {folder}')
        return True
    if needs_translation_refresh(target_language):
        logger.info(f'先寫配音大綱再翻譯：{folder}')
        ensure_episode_bible(
            folder, _load_folder_info(folder), _source_transcript(folder),
            target_language, method,
        )
    translation_path = os.path.join(folder, 'translation.json')
    if os.path.isfile(translation_path) and _same_lang(
        load_dub_meta(folder).get('translation'), target_language, 'translation'
    ):
        if needs_translation_refresh(target_language):
            logger.info(f'依大綱從原文重翻：{folder}')
            return refresh_translation_lines(folder, target_language, method)
        logger.info(f'收緊譯文字數以塞進原句槽位：{folder}')
        return tighten_overlong_lines(folder, target_language, method)
    if os.path.exists(translation_path) and not _same_lang(
        load_dub_meta(folder).get('translation'), target_language, 'translation'
    ):
        logger.info(f'改翻 {target_language}，其他語言版本留在 translations/：{folder}')
        from tools.target_language import clear_tts_cache
        os.remove(translation_path)
        clear_tts_cache(folder)
    
    info = _load_folder_info(folder)
    transcript_path = os.path.join(folder, 'transcript.json')
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    transcript = _prepare_source_lines(transcript, target_language, folder=folder)
    summary = ensure_episode_bible(folder, info, transcript, target_language, method)
    if not summary:
        logger.error(f'Failed to summarize {folder}')
        return False

    translation_path = os.path.join(folder, 'translation.json')
    translation = _translate(summary, transcript, target_language, method)
    for i, line in enumerate(transcript):
        raw = translation[i] if i < len(translation) else ''
        if _forbids_chinese_script(target_language):
            line['translation'] = _usable_target_text(raw, target_language)
        else:
            line['translation'] = raw
    transcript = split_sentences(transcript, target_language=target_language)
    _save_transcript(folder, transcript, target_language)
    _review_after_write(folder, target_language, method)
    return summary, transcript


def _review_after_write(folder, target_language, method):
    try:
        from tools.cost_tracker import mark_stage
        from tools.translation_review import apply_selected_review, auto_apply_review_picks, review_folder
        mark_stage('字幕審核修改...')
        report = review_folder(folder, target_language, method=method)
        n = len((report or {}).get('findings') or [])
        picks = auto_apply_review_picks(folder, target_language)
        if picks:
            _, applied, skipped, message, _ = apply_selected_review(
                folder, target_language, picks,
            )
            logger.info(f'審稿標出 {n} 句，已直接改譯文：{message}')
        elif n:
            logger.info(f'審稿標出 {n} 句，沒有可套用的建議譯文：{folder}')
        else:
            logger.info(f'審稿未標異常：{folder}')
        tighten_overlong_lines(folder, target_language, method=method)
    except Exception as exc:
        from tools.job_control import JobStopped
        if isinstance(exc, JobStopped):
            raise
        logger.warning(f'審稿略過：{exc}')


def translate_all_transcript_under_folder(folder, method, target_language):
    from tools.target_language import translation_cache_ok
    summary_json , translate_json = None, None
    for root, dirs, files in os.walk(folder):
        dirs[:] = [name for name in dirs if name not in {'translations', 'wavs', 'SPEAKER', '__pycache__'}]
        if 'transcript.json' not in files:
            continue
        if translation_cache_ok(root, target_language):
            if _folder_has_leftover_chinese(root, target_language):
                logger.info(f'譯文殘留中文，只重翻那些句：{root}')
                summary_json, translate_json = refresh_leftover_chinese_lines(root, target_language, method)
            else:
                fill_changed_source_translations(root, target_language, method)
                summary_json = json.load(open(os.path.join(root, 'summary.json'), 'r', encoding='utf-8'))
                translate_json = json.load(open(os.path.join(root, 'translation.json'), 'r', encoding='utf-8'))
                logger.info(f'Translation already exists in {root}')
        else:
            summary_json , translate_json = translate(method, root, target_language)
    return f'Translated all videos under {folder}',summary_json , translate_json

if __name__ == '__main__':
    # translate_all_transcript_under_folder(r'videos', 'LLM' , '简体中文')
    # translate_all_transcript_under_folder(r'videos', 'OpenAI' , '简体中文')
    translate_all_transcript_under_folder(r'videos', 'ernie' , '简体中文')