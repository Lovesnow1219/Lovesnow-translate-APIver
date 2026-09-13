# -*- coding: utf-8 -*-
"""Few-shot examples and system prompts for dubbing translation."""
import os

from tools.target_language import is_chinese_target, translation_language
from tools.translation_bible import bible_context

_STYLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prompts', 'dubbing_style.md'))


def _house_review_rules():
    """Load an editable creative brief, never agent rules or prior-episode fixes."""
    try:
        with open(_STYLE_PATH, encoding='utf-8') as handle:
            return handle.read().strip()
    except OSError:
        return ''


DUBBING_SHOTS = {
    'English': (
        ('Translate the whole line into natural spoken English:"等一下，我还没说完。"', "Wait, I'm not done yet."),
        ('Translate the whole line into natural spoken English:"你说会帮忙的。那你倒是动啊！"', "You said you'd help. So move!"),
        ('Translate the whole line into natural spoken English:"要是赶不上呢？"', "What if we're too late?"),
    ),
    'Vietnamese': (
        ('Translate the whole line into spoken Vietnamese in 1.2s, up to 6 syllables (one space per syllable). Keep every clause; speak numbers; do not end with … unless the source is unfinished:"知识就是力量。"', 'Tri thức là sức mạnh.'),
        ('Translate the whole line into spoken Vietnamese in 2.0s, up to 10 syllables (one space per syllable). Keep every clause; speak numbers; do not end with … unless the source is unfinished:"生存还是毁灭，这是一个值得考虑的问题。"', 'Sống hay chết, đó mới là vấn đề.'),
        ('Translate the whole line into spoken Vietnamese in 2.0s, up to 12 syllables (one space per syllable). Keep every clause; speak numbers:"目前进度是十分之零。"', 'Tiến độ giờ là không trên mười.'),
        ('Translate into spoken Vietnamese, about 0.5s:"呵。"', 'Hề.'),
        ('Translate into spoken Vietnamese, about 0.5s:"哼。"', 'Hừ.'),
        ('Translate into spoken Vietnamese, about 0.6s:"嘿嘿。"', 'He he.'),
        ('Translate into spoken Vietnamese, about 0.6s:"嗯。"', 'Ừ.'),
    ),
    'Thai': (
        ('Translate in 1.2s:"知识就是力量。"', 'ความรู้คือพลัง'),
        ('Translate in 2.0s:"生存还是毁灭，这是一个值得考虑的问题。"', 'จะอยู่หรือตาย นั่นคือคำถาม'),
    ),
    'Indonesian': (
        ('Translate in 1.2s, max 4 words:"知识就是力量。"', 'Pengetahuan adalah kekuatan.'),
        ('Translate in 2.0s, max 8 words:"生存还是毁灭，这是一个值得考虑的问题。"', 'Hidup atau mati? Itu pertanyaannya.'),
    ),
    'Malay': (
        ('Translate in 1.2s, max 4 words:"知识就是力量。"', 'Ilmu itu kuasa.'),
        ('Translate in 2.0s, max 8 words:"生存还是毁灭，这是一个值得考虑的问题。"', 'Hidup atau mati? Itulah soalan.'),
    ),
    'Filipino': (
        ('Translate in 1.2s, max 5 words:"知识就是力量。"', 'Ang kaalaman ay kapangyarihan.'),
        ('Translate in 2.0s, max 8 words:"生存还是毁灭，这是一个值得考虑的问题。"', 'Mabuhay o mamatay? Iyon ang tanong.'),
    ),
    'Spanish': (
        ('Translate in 1.2s, max 5 words:"知识就是力量。"', 'El saber es poder.'),
        ('Translate in 2.0s, max 8 words:"生存还是毁灭，这是一个值得考虑的问题。"', '¿Vivir o morir? Esa es la cuestión.'),
    ),
    'French': (
        ('Translate in 1.2s, max 5 words:"知识就是力量。"', "Le savoir, c'est le pouvoir."),
        ('Translate in 2.0s, max 8 words:"生存还是毁灭，这是一个值得考虑的问题。"', "Vivre ou mourir ? Voilà la question."),
    ),
    'Japanese': (
        ('Translate the whole line into spoken Japanese in 1.2s, up to 12 characters. Keep particles; write invented names in katakana:"知识就是力量。"', '知識は力だ。'),
        ('Translate the whole line into spoken Japanese in 2.0s, up to 20 characters. Keep particles; write invented names in katakana:"生存还是毁灭，这是一个值得考虑的问题。"', '生きるか死ぬか、それが問題だ。'),
        ('Translate the whole line into spoken Japanese in 2.0s, up to 20 characters. Keep particles; write invented names in katakana:"目前进度是十分之零。"', '今の進捗はゼロの十だぞ。'),
        ('Translate into spoken Japanese, about 1.8s:"我带你去见见世面。"', '世の中を見せてやる。'),
        ('Translate into spoken Japanese, about 0.5s:"呵。"', 'ふふ。'),
        ('Translate into spoken Japanese, about 0.5s:"哼。"', 'ふん。'),
        ('Translate into spoken Japanese, about 0.6s:"嘿嘿。"', 'へへ。'),
        ('Translate into spoken Japanese, about 0.6s:"嗯。"', 'ん。'),
    ),
    'Korean': (
        ('Translate in 1.2s:"知识就是力量。"', '아는 것이 힘이다.'),
        ('Translate in 2.0s:"生存还是毁灭，这是一个值得考虑的问题。"', '사느냐 죽느냐, 그것이 문제로다.'),
    ),
    'Cantonese': (
        ('Translate in 1.2s:"知识就是力量。"', '知識就係力量。'),
        ('Translate in 2.0s:"生存还是毁灭，这是一个值得考虑的问题。"', '生存定毀滅，呢個先係問題。'),
    ),
}


def dubbing_fixed_message(summary, target_language='简体中文'):
    info = bible_context(summary) or f'This is a video called "{summary.get("title")}". {summary.get("summary")}.'
    lang = translation_language(target_language)
    if is_chinese_target(target_language):
        return [
            {'role': 'system', 'content': f'You are an expert in the field of this video.\n{info}\nTranslate the sentence into {target_language}. 下面我让你来充当翻译家，你的目标是把任何语言翻译成{target_language}，请翻译时不要带翻译腔，而是要翻译得自然、流畅和地道，使用优美和高雅的表达方式。请将人工智能的“agent”翻译为“智能体”，强化学习中是`Q-Learning`而不是`Queue Learning`。数学公式写成plain text，不要使用latex。确保翻译正确和简洁。注意信达雅。'},
            {'role': 'user', 'content': f'使用地道的{target_language}Translate:"Knowledge is power."'},
            {'role': 'assistant', 'content': '翻译：“知识就是力量。”'},
            {'role': 'user', 'content': f'使用地道的{target_language}Translate:"To be or not to be, that is the question."'},
            {'role': 'assistant', 'content': '翻译：“生存还是毁灭，这是一个值得考虑的问题。”'},
        ]
    shots = DUBBING_SHOTS.get(lang) or (
        (f'Translate into spoken {lang}:"知识就是力量。"', 'Knowledge is power.'),
    )
    stats = 'Game stats and lists must be compact (e.g. +15%/45%, HP 12). '
    if lang == 'Vietnamese':
        completeness = (
            'Write spoken Vietnamese, not telegram fragments. '
            'Keep the subject, object, condition, and ending. '
            'Speak numbers: không trên mười, ba trăm, cộng mười lăm phần trăm. Never write 0/10, %, or +. '
            'Do not leave Chinese characters. Do not telegraph or drop the last clause. '
            'Keep spoken flavor: 嗯/恩 → Ừ; 呵 → Hề; 哼 → Hừ; 嘿嘿 → He he. Do not drop a particle-only line. '
            'Do not end with an ellipsis unless the source is unfinished. '
        )
        stats = 'Speak game stats in Vietnamese words, never +15% or 0/10. '
    elif lang == 'English':
        completeness = (
            'Write idiomatic spoken English an actor can say naturally. '
            'Preserve who does what to whom, questions, negation, conditions, and unfinished speech. '
            'Use everyday verbs and natural contractions when the character is casual; '
            'use a formal register only when the source character or creative brief calls for it. '
            'Keep the setup and punchline in their original order. '
            'Do not satisfy a word budget by breaking grammar, changing the action, or deleting a clause. '
            'Prefer a slightly longer faithful line if no natural shorter version exists. '
            'Prefer direct verbs to noun-heavy explanations and weak verb phrases when '
            'they express the same action. Recast the sentence naturally instead of copying '
            'Chinese clause structure word for word. '
            'Use confirmed proper names consistently; ordinary roles and unnamed organizations '
            'are not proper names and should use normal English capitalization. '
            'Resolve ambiguous words through the current scene, never a genre assumption. '
            'Keep brief vocal reactions and the speaker’s emotional intent. '
        )
        stats = 'Render numbers and any on-screen stats in a form that sounds natural when spoken. '
    elif lang == 'Japanese':
        completeness = (
            'Write spoken Japanese a viewer would actually hear. '
            'Do not calque Chinese compounds into kanji stacks that sound like Mandarin. '
            'Use the outline glossary for people and invented terms. Write those names in katakana. '
            'Never leave Chinese characters for names. '
            'Never spell 音読み in hiragana. '
            'Keep everyday mixed Japanese with は/が/を/だ/ぞ. '
            'Never write 0/10, %, or +. '
            'Keep 嗯 → ん, 呵 → ふふ, 哼 → ふん, 嘿嘿 → へへ. '
            'Keep the subject, object, condition, and ending. '
            'Do not end with an ellipsis unless the source is unfinished. '
        )
        stats = 'Speak game stats in Japanese words, never +15% or 0/10. '
    else:
        completeness = (
            f'Prefer short conversational {lang} over literal translation. '
            'Cut filler, repetition, and written-style clauses. '
        )
    fixed_message = [
        {'role': 'system', 'content': (
            f'You are a professional {lang} dubbing writer.\n'
            f'{info}\n'
            f'Write spoken dialogue in {lang} that can be said in about the given time. '
            f'{completeness}'
            f'{review_rewrite_rules(target_language)}'
            f'{stats}'
            'Keep names, numbers, and the same meaning. '
            'Do not add a clause the source does not speak. '
            "Keep the speaker's attitude: sarcasm, anger, fear, relief, surprise, or a laugh. "
            'If the source laughs (哈哈/呵呵), keep a short natural laugh in the target language. '
            'Do not write stage directions or [tags]. '
            f'Output ONLY the dubbed {lang} line. No quotes, labels, pinyin, romanization notes, or explanations.'
        )},
    ]
    for user, assistant in shots:
        fixed_message.append({'role': 'user', 'content': user})
        fixed_message.append({'role': 'assistant', 'content': assistant})
    return fixed_message


def review_rewrite_rules(target_language='English'):
    """Contextual editing criteria; no fixed meanings from previous episodes."""
    if is_chinese_target(target_language):
        return ''
    lang = translation_language(target_language)
    house = _house_review_rules()
    prefix = f'General dubbing guidance (adapt to this episode): {house} ' if house else ''
    return prefix + (
        f'Edit as a native {lang} dialogue writer. Preserve meaning, speaker intent, '
        'addressee, negation, conditions, and important names. '
        'Judge idiomatic phrasing and register as well as literal accuracy. '
        'Use everyday phrasing for casual dialogue, unless the source performance is deliberately formal. '
        'Preserve the relationship between a comic setup, the reply, and the punchline. '
        'Shorten redundancy before substance. A small timing overrun is preferable to broken '
        'grammar or a missing thought; if no natural shorter line exists, leave suggest empty. '
        'The old English wording is not a template. Recast entire clauses with direct verbs '
        'and idiomatic phrasing when this saves meaningful spoken time without losing facts. '
        'Preserve each concrete actor and action, parallel contrasts, and the exact logic '
        'of alternatives, thresholds and conditions. Do not replace them with vague abstractions. '
        'Keep unfinished speech unfinished and vocal reactions intact. '
        'Do not shorten a line solely because an earlier line delayed its start. '
        'If a card appears to contain different speakers, flag the segmentation issue; '
        'do not silently discard part of the source or invent a speaker assignment. '
        'Resolve ambiguous words from nearby dialogue and the current episode’s glossary. '
        'Do not impose a setting or a fixed translation learned from another story. '
    )


def review_editor_brief(target_language='English'):
    if is_chinese_target(target_language):
        return ''
    return review_rewrite_rules(target_language) + (
        'During the full-dialogue review, flag missing or invented meaning, unnatural collocations, '
        'unmotivated register changes, inconsistent proper names, damaged questions, '
        'and lost comic timing even when a safe rewrite is unavailable. '
        'Use measured audio duration when supplied; word counts are only an estimate. '
        'Distinguish actual speech duration from delay inherited from a previous line. '
        'Flag suspected mixed speakers as GLUE with no deletion-based rewrite. '
        'For each safe suggestion, provide the complete natural replacement line. '
        'Do not turn a diagnostic flag into an unsupported change of meaning. '
    )


def review_system_preamble(target_language='English'):
    """Shared reviewer identity for every review stage."""
    if is_chinese_target(target_language):
        return ''
    return (
        'You are an experienced dubbing editor. Judge the full spoken exchange, '
        'including meaning, idiomatic delivery, character intent, and timing. '
        'The operator reads Traditional Chinese. '
        + review_editor_brief(target_language)
    )
