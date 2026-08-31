# -*- coding: utf-8 -*-
"""Few-shot examples and system prompts for dubbing translation."""
from tools.target_language import is_chinese_target, translation_language
from tools.translation_bible import bible_context

DUBBING_SHOTS = {
    'English': (
        ('Translate into spoken English, about 1.2s. A bit over is OK:"知识就是力量。"', 'Knowledge is power.'),
        ('Translate into spoken English, about 2.0s. A bit over is OK:"生存还是毁灭，这是一个值得考虑的问题。"', "Live or die — that's the question."),
        ('Translate into spoken English, about 2.0s. A bit over is OK:"目前投入领营十分之零。"', "The camp's input is still zero out of ten."),
        ('Translate into spoken English, about 1.5s. A bit over is OK:"必须抓住这个机会啊。"', 'Gotta take this shot!'),
        ('Translate into spoken English, about 1.8s. A bit over is OK:"威拉带你去见见世面。"', "Willa, I'll show you the world."),
        ('Translate into spoken English, about 0.5s. A bit over is OK:"呵。"', 'Heh.'),
        ('Translate into spoken English, about 0.5s. A bit over is OK:"哼。"', 'Hmph.'),
        ('Translate into spoken English, about 0.6s. A bit over is OK:"嘿嘿。"', 'Hehe.'),
        ('Translate into spoken English, about 0.6s. A bit over is OK:"嗯。"', 'Hmm.'),
        ('Translate into spoken English, about 0.8s. A bit over is OK:"好看，买！"', 'Cute. Sold!'),
        ('Translate into spoken English, about 0.8s. A bit over is OK:"好看，好看。"', 'Nice. Nice.'),
        ('Translate into spoken English, about 1.4s. A bit over is OK:"本小姐果然穿什么都好看。"', 'Of course I look good in anything!'),
        ('Translate into spoken English, about 1.8s. A bit over is OK:"嗯,这套是挺不错的。"', "Hmm, this one's pretty nice."),
    ),
    'Vietnamese': (
        ('Translate the whole line into spoken Vietnamese in 1.2s, up to 6 syllables (one space per syllable). Keep every clause; speak numbers; do not end with … unless the source is unfinished:"知识就是力量。"', 'Tri thức là sức mạnh.'),
        ('Translate the whole line into spoken Vietnamese in 2.0s, up to 10 syllables (one space per syllable). Keep every clause; speak numbers; do not end with … unless the source is unfinished:"生存还是毁灭，这是一个值得考虑的问题。"', 'Sống hay chết, đó mới là vấn đề.'),
        ('Translate the whole line into spoken Vietnamese in 2.0s, up to 12 syllables (one space per syllable). Keep every clause; speak numbers:"目前投入领营十分之零。"', 'Doanh trại giờ là không trên mười.'),
        ('Translate into spoken Vietnamese, about 0.5s:"呵。"', 'Hề.'),
        ('Translate into spoken Vietnamese, about 0.5s:"哼。"', 'Hừ.'),
        ('Translate into spoken Vietnamese, about 0.6s:"嘿嘿。"', 'He he.'),
        ('Translate into spoken Vietnamese, about 0.6s:"嗯。"', 'Ừ.'),
        ('Translate into spoken Vietnamese, about 0.8s:"好看，买！"', 'Đẹp. Mua!'),
        ('Translate into spoken Vietnamese, about 0.8s:"好看，好看。"', 'Đẹp. Đẹp.'),
        ('Translate into spoken Vietnamese, about 1.4s:"本小姐果然穿什么都好看。"', 'Mặc gì tôi cũng đẹp!'),
        ('Translate into spoken Vietnamese, about 1.8s:"嗯,这套是挺不错的。"', 'Ừ, bộ này cũng ổn đó.'),
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
        ('Translate the whole line into spoken Japanese in 1.2s, up to 12 characters. Keep particles; write invented names in katakana; keep 99/30 as digits:"知识就是力量。"', '知識は力だ。'),
        ('Translate the whole line into spoken Japanese in 2.0s, up to 20 characters. Keep particles; write invented names in katakana; keep 99/30 as digits:"生存还是毁灭，这是一个值得考虑的问题。"', '生きるか死ぬか、それが問題だ。'),
        ('Translate the whole line into spoken Japanese in 2.0s, up to 20 characters. Keep particles; write invented names in katakana:"目前投入领营十分之零。"', '今の領営はゼロの十だぞ。'),
        ('Translate into spoken Japanese, about 1.8s:"威拉带你去见见世面。"', 'ウィラ、世の中を見せてやる。'),
        ('Translate into spoken Japanese, about 1.6s:"现实世界货币转化卡"', 'リアルワールドのマネーカード'),
        ('Translate into spoken Japanese, about 0.5s:"呵。"', 'ふふ。'),
        ('Translate into spoken Japanese, about 0.5s:"哼。"', 'ふん。'),
        ('Translate into spoken Japanese, about 0.6s:"嘿嘿。"', 'へへ。'),
        ('Translate into spoken Japanese, about 0.6s:"嗯。"', 'ん。'),
        ('Translate into spoken Japanese, about 0.8s:"好看，买！"', 'いいね、買う！'),
        ('Translate into spoken Japanese, about 0.8s:"好看，好看。"', 'いいね、いいね。'),
        ('Translate into spoken Japanese, about 1.4s:"本小姐果然穿什么都好看。"', 'この私、何着ても似合うんだから！'),
        ('Translate into spoken Japanese, about 1.8s:"嗯,这套是挺不错的。"', 'ん、これ結構いいね。'),
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
            '本小姐 is the speaker (tôi). Repeated 好看 is Đẹp. Đẹp. not a long compliment. '
            'Do not end with an ellipsis unless the source is unfinished. '
        )
        stats = 'Speak game stats in Vietnamese words, never +15% or 0/10. '
    elif lang == 'English':
        completeness = (
            'Write spoken game/anime English a viewer would actually hear, not textbook and not chopped fragments. '
            'Use the outline name list exactly (same spelling every time). '
            'Keep the subject, object, condition, and ending. '
            'Keep spoken flavor: 啊/呀/吧/呢/嘛/了吗 → huh/yeah/right; 嗯/恩 → Hmm or Mm; 呵 → Heh; 哼 → Hmph; 竟然 → even; 必须 → gotta; 可是/真是 → but/man. '
            '本小姐 is the speaker: I. Repeated 好看 is Nice. Nice. or Looks good—not a long “beautiful, just beautiful” line. '
            'If the source is only 嗯 or 恩, output Hmm. or Mm. If it is only 呵, output Heh. If it is only 哼, output Hmph. If it is 嘿嘿, output Hehe. Do not drop it. '
            'A name at the start of the Chinese line is usually the listener (vocative), not the subject. '
            '你家领主我 means the speaker: I, your lord. Do not add names that are not in this line. '
            'If the line is about food or drink, 绝配 means they pair perfectly, not a couple. '
            'Do not turn a statement into a question unless the source asks. '
            'Narration, system panels, and inner monologue may run a little longer than the slot. '
            'Never output a stump like "Congratulations." or "You see?" unless that is the whole source. '
            'Write invented names so TTS can say them. Do not leave Chinese. '
            'Keep insults and emotion in the wording. '
        )
        stats = 'Game stats may stay compact if they still sound spoken. '
    elif lang == 'Japanese':
        completeness = (
            'Write spoken anime/game Japanese a seiyuu would say. '
            'Do not calque Chinese compounds (現実世界貨幣変換、戦争ゲーム、合法的に、人為的に). '
            'Those sound like Mandarin when dubbed. Use 和語 or katakana: '
            '现实世界 → リアルワールド / こっちの世界; 货币转化卡 → マネーカード; '
            '战争游戏 → ウォーゲーム; 国库 → 金庫; 快乐水 → コーラ. '
            'Never spell 音読み in hiragana (forbidden: こっこ, きんか, てんもんがくてき). '
            'Keep everyday mixed Japanese (船、金、故郷) with は/が/を/だ/ぞ. '
            'People: 威拉/薇拉/维拉 → ウィラ only. Never leave Chinese names. 本小姐 is この私. '
            '横着走 is のし歩く, not 横向き. '
            'Keep LV99, 30, 640 as digits (レベル99, 30日). Never write 0/10, %, or +. '
            'Keep 嗯 → ん, 呵 → ふふ, 哼 → ふん, 嘿嘿 → へへ. Repeated 好看 is いいね、いいね。 '
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
            f'{stats}'
            'Keep names, numbers, and the same meaning. '
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

