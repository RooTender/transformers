<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Tokenizery

Tokenizer jest odpowiedzialny za przygotowanie danych wejściowych dla modelu. Biblioteka zawiera tokenizery dla wszystkich modeli. Większość tokenizerów jest dostępna w dwóch wersjach: pełnej implementacji w Pythonie i "Fast" (pl. *szybkiej*) implementacji opartej na bibliotece Rust [🤗 Tokenizers](https://github.com/huggingface/tokenizers). "Fast" implementacje pozwalają na:

1. znaczne przyspieszenie, w szczególności podczas tokenizacji wsadowej,
2. dodatkowe metody mapowania między oryginalnym ciągiem znaków (znakami i słowami) a przestrzenią tokenów (np. uzyskanie indeksu tokena zawierającego dany znak lub zakres znaków odpowiadających danemu tokenowi).

Klasy bazowe [`PreTrainedTokenizer`] i [`PreTrainedTokenizerFast`] implementują wspólne metody kodowania ciągów znaków w danych wejściowych modelu (patrz poniżej) oraz instancjonowania/zapisywania tokenizerów pythonowskich i "Fast" na bazie danych z lokalnego pliku, katalogu albo z wstępnie wytrenowanego tokenizera dostarczonego przez bibliotekę (pobranej z repozytorium AWS S3 firmy HuggingFace). Oba opierają się na [`~tokenization_utils_base.PreTrainedTokenizerBase`], który zawiera zwykłe metody, oraz [`~tokenization_utils_base.SpecialTokensMixin`].

Zatem [`PreTrainedTokenizer`] i [`PreTrainedTokenizerFast`] implementują główne metody korzystania ze wszystkich tokenizerów:

- Tokenizacja (dzielenie ciągów znaków na ciągi tokenów pod-słów), konwersja ciągów tokenów na identyfikatory i z powrotem oraz kodowanie/dekodowanie (tj. tokenizacja i konwersja na liczby całkowite).
- Dodawanie nowych tokenów do słownika w sposób niezależny od struktury bazowej (BPE, SentencePiece...).
- Zarządzanie specjalnymi tokenami (takimi jak maska, początek zdania itp.): dodawanie ich, przypisywanie ich do atrybutów w tokenizatorze w celu łatwego dostępu i upewnianie się, że nie zostaną podzielone podczas tokenizacji.

[`BatchEncoding`] przechowuje dane wyjściowe metod kodowania [`~tokenization_utils_base.PreTrainedTokenizerBase`] (`__call__`, `encode_plus` i `batch_encode_plus`) i pochodzi ze słownika Pythona. Kiedy tokenizator jest tokenizatorem czysto pythonowym, klasa ta zachowuje się jak standardowy słownik pythonowy i przechowuje różne dane wejściowe modelu obliczone przez te metody (`input_ids`, `attention_mask`...). Gdy tokenizer jest "Fast" tokenizerem (tj. wspieranym przez HuggingFace [biblioteka tokenizerów](https://github.com/huggingface/tokenizers)), klasa ta zapewnia dodatkowo kilka zaawansowanych metod dopasowania, które można wykorzystać do mapowania między oryginalnym ciągiem znaków (znakami i słowami) a przestrzenią tokenów (np. uzyskiwanie indeksu tokenu zawierającego dany znak lub zakresu znaków odpowiadających danemu tokenowi).


## PreTrainedTokenizer

[[autodoc]] PreTrainedTokenizer
    - __call__
    - add_tokens
    - add_special_tokens
    - apply_chat_template
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## PreTrainedTokenizerFast

[`PreTrainedTokenizerFast`] zależą od biblioteki [tokenizers](https://huggingface.co/docs/tokenizers). Tokenizery uzyskane z biblioteki 🤗 Tokenizers, którą można bardzo łatwo wczytać do 🤗 Transformers. Zapoznaj się ze stroną [Używanie tokenizerów z 🤗 Tokenizers](../fast_tokenizers) aby zrozumieć, jak to się robi.

[[autodoc]] PreTrainedTokenizerFast
    - __call__
    - add_tokens
    - add_special_tokens
    - apply_chat_template
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## BatchEncoding

[[autodoc]] BatchEncoding
