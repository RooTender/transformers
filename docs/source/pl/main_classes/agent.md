<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Agents & Tools

<Tip warning={true}>

Transformers Agents to eksperymentalny interfejs API, który może ulec zmianie w dowolnym momencie. Wyniki zwracane przez agentów mogą się różnić, ponieważ API lub modele bazowe są podatne na zmiany.

</Tip>

Aby dowiedzieć się więcej o agentach i narzędziach, przeczytaj [przewodnik wprowadzający](../transformers_agents). Ta strona zawiera dokumentację API dla klas bazowych.

## Agents

Oferujemy trzy typy agentów: [`HfAgent`] wykorzystuje endpointy wnioskowania dla modeli opensource, [`LocalAgent`] wykorzystuje lokalnie zainstalowany model, a [`OpenAiAgent`] wykorzystuje modele zamknięte od OpenAI.

### HfAgent

[[autodoc]] HfAgent

### LocalAgent

[[autodoc]] LocalAgent

### OpenAiAgent

[[autodoc]] OpenAiAgent

### AzureOpenAiAgent

[[autodoc]] AzureOpenAiAgent

### Agent

[[autodoc]] Agent
    - chat
    - run
    - prepare_for_new_chat

## Tools

### load_tool

[[autodoc]] load_tool

### Tool

[[autodoc]] Tool

### PipelineTool

[[autodoc]] PipelineTool

### RemoteTool

[[autodoc]] RemoteTool

### launch_gradio_demo

[[autodoc]] launch_gradio_demo

## Agent Types

Agenci mogą obsługiwać dowolny typ obiektu pomiędzy narzędziami; narzędzia, będąc całkowicie multimodalne, mogą przyjmować i zwracać tekst, obraz, dźwięk, wideo i inne typy. Aby zwiększyć kompatybilność między narzędziami, a także poprawnie renderować te wyniki w ipythonie (jupyter, colab, ipython notebooks, ...), implementujemy wrappery wokół tych typów.

Obiekty, na które nałożono wrapper, powinny nadal zachowywać się jak na początku; obiekt tekstowy powinien nadal zachowywać się jak ciąg znaków, obiekt obrazu powinien nadal zachowywać się jak `PIL.Image`.

Typy te mają trzy konkretne cele:

- Wywołanie `to_raw` na typie powinno zwrócić obiekt bazowy
- Wywołanie `to_string` na typie powinno zwrócić obiekt jako ciąg znaków: może to być ciąg znaków w przypadku `AgentText`, ale w innych przypadkach będzie to ścieżka serializowanej wersji obiektu.
- Wyświetlenie go w kernelu ipythona powinno poprawnie wyświetlić obiekt

### AgentText

[[autodoc]] transformers.tools.agent_types.AgentText

### AgentImage

[[autodoc]] transformers.tools.agent_types.AgentImage

### AgentAudio

[[autodoc]] transformers.tools.agent_types.AgentAudio
