# -*- coding: utf-8 -*-
"""Few-shot examples and system prompts for dubbing translation."""
import os

from tools.target_language import is_chinese_target, translation_language
from tools.translation_bible import bible_context

_HOUSE_RULE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '.cursor', 'rules', 'dubbing-pipeline.mdc',
))


def _house_section(raw, title):
    token = f'## {title}'
    if token not in raw:
        return ''
    body = raw.split(token, 1)[1]
    if '\n## ' in body:
        body = body.split('\n## ', 1)[0]
    bullets = []
    for line in body.splitlines():
        text = line.strip()
        if text.startswith('- '):
            bullets.append(text[2:])
    return ' '.join(bullets)


def _house_review_rules():
    """Same 時軸／譯文節奏 bullets the desktop agent sees."""
    try:
        raw = open(_HOUSE_RULE_PATH, encoding='utf-8').read()
    except OSError:
        return ''
    parts = [_house_section(raw, title) for title in ('時軸', '譯文節奏')]
    return ' '.join(part for part in parts if part)

DUBBING_SHOTS = {
    'English': (
        ('Translate the whole line into spoken English in 1.2s, max 3 words:"知识就是力量。"', 'Knowledge is power.'),
        ('Translate the whole line into spoken English in 2.0s, max 6 words:"生存还是毁灭，这是一个值得考虑的问题。"', "Live or die — that's the question."),
        ('Translate the whole line into spoken English in 2.0s, max 6 words:"目前进度是十分之零。"', "Progress is still zero of ten."),
        ('Translate the whole line into spoken English in 1.5s, max 4 words:"必须抓住这个机会啊。"', 'Gotta take this shot!'),
        ('Translate the whole line into spoken English in 1.0s, max 3 words:"滚开！"', 'Get out!'),
        ('Translate the whole line into spoken English in 1.5s, max 5 words:"你吓死我了。"', 'You scared me to death.'),
        ('Translate the whole line into spoken English in 1.8s, max 5 words:"我带你去见见世面。"', "I'll show you the world."),
        ('Translate the whole line into spoken English in 0.5s, max 2 words:"呵。"', 'Heh.'),
        ('Translate the whole line into spoken English in 0.5s, max 2 words:"哼。"', 'Hmph.'),
        ('Translate the whole line into spoken English in 0.6s, max 2 words:"嘿嘿。"', 'Hehe.'),
        ('Translate the whole line into spoken English in 0.6s, max 2 words:"嗯。"', 'Hmm.'),
        ('Translate the whole line into spoken English in 1.6s, max 8 words:"先这样，等我离开这里……"', 'For now—once I leave here...'),
        ('Translate the whole line into spoken English in 2.0s, max 6 words:"好啊，那我就告诉你们吧。卫兵，上！"', "Fine, I'll tell you."),
        ('Translate the whole line into spoken English in 1.2s, max 4 words:"是、是……不！"', 'Y-yes... no!'),
        ('Translate the whole line into spoken English in 1.8s, max 6 words:"老头，你去过那里吗？"', 'Geezer, you been there?'),
        ('Translate the whole line into spoken English in 1.5s, max 6 words:"你拿枪指着我？"', "You're pointing that spear at me?"),
        ('Translate the whole line into spoken English in 1.8s, max 7 words:"空气中没有灵气。"', 'No spiritual energy in the air.'),
        ('Translate the whole line into spoken English in 2.0s, max 12 words:"称此地俊杰无愧于天地。"', 'Hero of this land, unashamed before Heaven and Earth!'),
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
            'Write spoken English a viewer would actually hear, like an actor, not a subtitle. '
            'Use the outline name list exactly (same spelling every time). '
            'Keep the subject, object, condition, and ending. '
            'Keep spoken flavor: 啊/呀/吧/呢/嘛/了吗 → huh/yeah/right; 嗯/恩 → Hmm or Mm; 呵 → Heh; 哼 → Hmph; 竟然 → even; 必须 → gotta; 可是/真是 → but/man. '
            'If the source is only 嗯 or 恩, output Hmm. or Mm. If it is only 呵, output Heh. If it is only 哼, output Hmph. If it is 嘿嘿, output Hehe. Do not drop it. '
            'A name at the start of the Chinese line is usually the listener (vocative), not the subject. '
            'Do not add names, clauses, or a call-to-action the Chinese does not say. '
            'Do not turn a statement into a question unless the source asks. '
            'If the Chinese is angry, scared, mocking, or desperate, the English must feel the same. '
            'Use contractions when the speaker is casual. Keep ！ as ! and ？ as ?. '
            'Do not polish anger or panic into polite English. '
            'When shortening, cut filler, not the person: keep the speech-act, addressee, and names. '
            'Questions stay questions. Laughs stay Hehe/Haha, not Ha! '
            'Keep technique and chant names; never join two mouths with " / ". '
            'Unfinished …… stays unfinished: keep the named array or art and trail off. '
            'Do not add somehow/anyway or close a guessed sentence. '
            'If one card glued two speakers (a reply then someone else\'s command), '
            'translate only the first mouth. '
            'If the line is already Latin letters or kana, copy it; do not clean it into proper English. '
            'Keep the verb in a question (seen/been/gone). Do not telegraph "X\'s depths?" '
            'A polearm 枪 is a spear, not a gun, unless the Chinese is a firearm. '
            'If ASR wrote 河道 next to 巔峰/强者, that is a realm, not a river. '
            'Heaven and Earth stay together; do not keep only Heaven. '
            '称〇俊傑 keeps the place; do not crush it to Hero. '
            'Do not stamp a glossary name onto a 3–8 character ASR hash that is not that name. '
            'Do not output a one-word bark unless the Chinese is also a bark. '
            'A slightly long line is better than broken English. '
            'Fit the original seconds. Narration may keep one extra beat, not a paragraph. '
            'Write invented names so TTS can say them. Do not leave Chinese. '
            'Keep insults and emotion in the wording. '
        )
        stats = 'Game stats may stay compact if they still sound spoken. '
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
    """Same editor brief as the desktop agent: house rules file + rewrite craft."""
    if is_chinese_target(target_language):
        return ''
    lang = translation_language(target_language)
    house = _house_review_rules()
    prefix = (
        f'Operator house rules (follow these; they are not optional): {house} '
        if house else ''
    )
    return prefix + (
        f'When you rewrite, edit like a human {lang} dubbing editor, not a compressor. '
        'Cut filler only (just, really, that, and then, unbelievably). '
        'Keep the speech-act: a question stays a question; a command stays a command; '
        'a boast stays a boast. '
        'Keep the addressee and spoken names already in the line. '
        'Keep laughs as laughs (嘿嘿→Hehe, 哈哈→Haha). Never replace them with Ha! '
        'Particle-only cards (嘿嘿/呵/哼/嗯) are never rewritten; the mixer fades them. '
        'Do not shorten the next spoken line just because a previous laugh made it late. '
        'Keep technique and chant names (the part after ·, 剑/诀/阵). Cut filler around them, not the name. '
        'Keep every clause of a poetic spell; do not crush it into a two-word compound. '
        'Never join two speakers with " / ", and never make one mouth speak the other\'s command. '
        'If a card glued two speakers, translate only the first mouth; leave suggest empty if you cannot split. '
        'Unfinished …… stays unfinished: keep the named thing and trail off. No somehow/anyway. '
        'A yes-then-no stammer (是……不) stays a stammer. Do not turn it into Fine, I\'ll tell you. '
        'Keep the verb in a question. A polearm 枪 is a spear, not a gun. '
        'A cultivation-realm homophone is not a river or channel. '
        '「空氣中沒有…」keeps the place (in the air / here). '
        '天地 is Heaven and Earth. 称〇俊傑 keeps the place name. '
        'If a Chinese episode line was recognized as English/Latin/kana, copy it. '
        'Do not clean it into proper English and do not invent another language. '
        'Empty 呸/高/嗯 on real speech get a short vocalization, not a blank. '
        'Never turn "How goes the fight?" into "Report." '
        'Never turn "How could this king lose?" into "Lose?" '
        'Never strip the name from "Die, Zhang San!" to "Die!" '
        'Never output broken English like "He dared eat." '
        'One-word barks only when the Chinese is also a bark (滚/死/上). '
        'A 0.2–0.5s overrun is better than a line that lost its person. '
        'If no natural shorter line exists, leave suggest empty. '
    )


def review_editor_brief(target_language='English'):
    """Full watch-through the desktop agent uses when judging a dubbed episode."""
    if is_chinese_target(target_language):
        return ''
    return review_rewrite_rules(target_language) + (
        'Watch-through — flag even if suggest stays empty: '
        'meaning lost versus the Chinese (dropped clause, name, addressee, or the ending); '
        'unfinished Chinese (……) must stay unfinished — keep the named array/art and trail off; '
        'do not add somehow/anyway or close a guessed sentence; '
        'a yes-then-no stammer must stay yes-then-no — do not rewrite it as a confession; '
        'do not telegraph a question by dropping the verb; '
        'polearm 枪 is spear, not gun; 河道巔峰/强者 is a realm homophone, not a waterway; '
        '「空氣中沒有…」must keep the place; 天地 is Heaven and Earth; '
        '称〇俊傑 keeps the place — do not crush it to Hero; '
        'do not stamp a glossary name onto ASR hash that is not that name; '
        'school, class, place, and title names stay intelligible — do not invent opaque number-codes; '
        'technique or chant names (·, 剑/诀/阵, or three-plus clauses) must keep the distinctive parts; '
        'two mouths must not share one dub: no " / ", and no one voice speaking the other mouth\'s order; '
        'if a card glued two speakers, flag GLUE and translate only the first mouth; '
        'Latin-script or kana ASR in a Chinese episode: copy as-is, mark FOREIGN, '
        'do not clean it into proper English and do not invent a third language; '
        'empty real speech (呸/高 and other 1–2 character spoken cards) gets a short vocalization; '
        'particle/laugh cards stay Hehe/Haha/Heh — never Ha!, and never shorten the next spoken line because a laugh ran long; '
        'questions stay questions; boasts stay boasts; one-word barks only when the Chinese is a bark; '
        'a 0.2–0.5s overrun is not a defect — do not rewrite to chase it; '
        'sung or credit lines: do not crush a song into a two-second bark; flag if the dub is far shorter than the picture; '
        'tone must match the Chinese (anger, fear, mockery, panic) — no polite textbook flatten; '
        'glossary names unused or respelt. '
        'If a natural spoken line exists, put the complete sentence in suggest. '
        'If it does not, leave suggest empty and still flag. '
    )


def review_system_preamble(target_language='English'):
    """Shared flagship-reviewer identity for every 審稿 stage."""
    if is_chinese_target(target_language):
        return ''
    return (
        'You are the same human dubbing editor as the desktop operator. '
        'This is flagship review: judge a watch-through, not a word-count compressor. '
        'House rules beat fitting the slot. '
        'The operator reads Traditional Chinese. '
        + review_editor_brief(target_language)
    )

