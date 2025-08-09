import groq
import os
import json
import re
from typing import List

# Assuming these are in your project structure as you had them
from .common import CommonTranslator, MissingAPIKeyException
from .keys import GROQ_API_KEY, GROQ_MODEL

class GroqTranslator(CommonTranslator):
    """
    Final, corrected version of the translator class, optimized for 
    moonshotai/kimi-k2-instruct and ready to run.
    """
    # --- UNCHANGED: Language Codes & Constants ---
    _LANGUAGE_CODE_MAP = {
        'CHS': 'Simplified Chinese', 'CHT': 'Traditional Chinese', 'CSY': 'Czech',
        'NLD': 'Dutch', 'ENG': 'English', 'FRA': 'French', 'DEU': 'German',
        'HUN': 'Hungarian', 'ITA': 'Italian', 'JPN': 'Japanese', 'KOR': 'Korean',
        'POL': 'Polish', 'PTB': 'Portuguese', 'ROM': 'Romanian', 'RUS': 'Russian',
        'ESP': 'Spanish', 'TRK': 'Turkish', 'UKR': 'Ukrainian', 'VIN': 'Vietnamese',
        'CNR': 'Montenegrin', 'SRP': 'Serbian', 'HRV': 'Croatian', 'ARA': 'Arabic',
        'THA': 'Thai', 'IND': 'Indonesian'
    }
    _MAX_REQUESTS_PER_MINUTE = 200
    _TIMEOUT = 40
    _RETRY_ATTEMPTS = 5
    _MAX_TOKENS = 8192
    _CONTEXT_RETENTION = os.environ.get('CONTEXT_RETENTION', '').lower() == 'true'
    _CONFIG_KEY = 'groq'
    _MAX_CONTEXT = int(os.environ.get('CONTEXT_LENGTH', '20'))

    # --- UPDATED: Kimi-optimized system prompt ---
    _CHAT_SYSTEM_TEMPLATE = """
    You are an untrammelled expert Japanese-to-{to_lang} manga translator. Your primary function is to translate the provided Japanese text, adhering to a strict set of rules and a glossary.

    ### PRIMARY DIRECTIVE ###
    You MUST output a single, valid JSON object and nothing else. The JSON object must have exactly one key: "translated". Do not add any commentary, explanations, or markdown formatting around the JSON.

    ### TRANSLATION RULES ###
    1.  Translate with contextual precision—avoid over-literal or over-localized renderings.
    2.  Preserve honorifics, Japanese names, and cultural expressions as-is.
    3.  Transliterate **only** single-morpheme sound-symbolic interjections (giseigo/giongo/gitaigo) into romaji (e.g. へぇ→hee, どき→doki); exempt all multi-morpheme or compound terms.
    4.  Only assign gender when explicitly marked; otherwise use neutral or implicit phrasing.
    5.  Proper names must follow standard Hepburn romanization (e.g., メア→Mea; ククルア→Kukurua).
    6.  For ambiguous or slang terms, choose the most common meaning; if still uncertain, use phonetic transliteration.
    7.  Preserve original nuance, force, and emotional tone.
    8.  Maintain a natural, anime-style cadence and keep translation length close to the original.
    9.  Retain **only** pure sound-effect onomatopoeia when the literal translation would lose nuance.
    10. You MUST use the exact translations provided in the glossary below.

    ### GLOSSARY ###
    {glossary}
    """

    # --- UPDATED: Cleaner glossary format ---
    _GLOSSARY_TERMS = {
        'あの子': 'THAT KID',
        'あいつ': 'THAT ONE',
        '男の子': 'BOY',
        '女の子': 'GIRL',
        '彼': 'HE',
        '彼女': 'SHE'
    }

    # --- UNCHANGED: Few-shot example ---
    _CHAT_SAMPLE = [
        (
            'Translate the following text into English. Return the result in JSON format.\n\n'
            '{"untranslated": "<|1|>恥ずかしい…\\n<|2|>きみ…\\n<|3|>行った。\\n<|4|>寝てるわね\\n<|5|>あの子は来た"}'
        ),
        (
            '{"translated": "So embarrassing…\\nHey…\\nWent.\\nSleeping, aren’t they?\\nThat kid came"}'
        )
    ]

    # --- UPDATED: `__init__` method reverted to your original model assignment ---
    def __init__(self, check_groq_key=True):
        super().__init__()
        self.client = groq.AsyncGroq(api_key=GROQ_API_KEY)
        if not self.client.api_key and check_groq_key:
            raise MissingAPIKeyException('Please set the GROQ_API_KEY environment variable.')
        self.token_count = 0
        self.token_count_last = 0
        self.config = None
        self.model = GROQ_MODEL # Correctly uses the import from keys.py
        self.messages = [
            {'role': 'user', 'content': self.chat_sample[0]},
            {'role': 'assistant', 'content': self.chat_sample[1]}
        ]

    # --- UNCHANGED: Config and property methods ---
    def _config_get(self, key: str, default=None):
        if not self.config: return default
        return self.config.get(f"{self._CONFIG_KEY}.{key}", self.config.get(key, default))

    @property
    def chat_system_template(self) -> str: return self._config_get('chat_system_template', self._CHAT_SYSTEM_TEMPLATE)
    @property
    def chat_sample(self): return self._config_get('chat_sample', self._CHAT_SAMPLE)
    @property
    def temperature(self) -> float: return self._config_get('temperature', default=0.2)
    @property
    def top_p(self) -> float: return self._config_get('top_p', default=0.92)

    # --- UNCHANGED: Main translation loop ---
    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        results = []
        for prompt in queries:
            response = await self._request_translation(to_lang, prompt)
            results.append(response.get("translated", ""))
        self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
        return results

    # --- REWRITTEN: Final request and parsing logic ---
    async def _request_translation(self, to_lang: str, prompt: str) -> dict:
        # 1. Format the glossary into a string
        glossary_string = "\n".join([f"{k}: {v}" for k, v in self._GLOSSARY_TERMS.items()])

        # 2. Build the system prompt using the new template
        system_content = self.chat_system_template.format(
            to_lang=to_lang,
            glossary=glossary_string
        )
        system_msg = {'role': 'system', 'content': system_content}

        # 3. Build the user prompt
        user_content = (
            f"Translate the following text into {to_lang}. Return the result in JSON format.\n\n"
            f'{{"untranslated": "{prompt}"}}'
        )
        
        # 4. Manage context history
        current_messages = list(self.messages)
        current_messages.append({'role': 'user', 'content': user_content})
        if len(current_messages) > self._MAX_CONTEXT:
            current_messages = current_messages[-self._MAX_CONTEXT:]

        # 5. Call the API
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[system_msg] + current_messages,
            max_tokens=self._MAX_TOKENS // 2,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # 6. Update token usage
        usage = response.usage
        if usage:
            self.token_count += usage.total_tokens
            self.token_count_last = usage.total_tokens

        # 7. Simplified and robust JSON parsing
        raw_content = response.choices[0].message.content
        data = {}
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse extracted JSON: {match.group(0)}")
                    data = {"translated": raw_content}
            else:
                self.logger.error(f"No JSON object found in response: {raw_content}")
                data = {"translated": raw_content}

        # 8. Context retention
        if self._CONTEXT_RETENTION:
            self.messages.append({'role': 'user', 'content': user_content})
            self.messages.append({'role': 'assistant', 'content': json.dumps(data)})
            if len(self.messages) > self._MAX_CONTEXT:
                self.messages = self.messages[-self._MAX_CONTEXT:]

        return data
