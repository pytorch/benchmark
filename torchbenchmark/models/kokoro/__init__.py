from argparse import Namespace
from pathlib import Path
from typing import Tuple

import torch
import re 
from torchbenchmark.tasks import OTHER

from ...util.model import BenchmarkModel

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from kokoro import KPipeline
from huggingface_hub import hf_hub_download


def load_single_voice(pipeline, voice: str):
    if voice in pipeline.voices:
        return pipeline.voices[voice]
    if voice.endswith('.pt'):
        f = voice
    else:
        f = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename=f'voices/{voice}.pt')
        assert voice.startswith(pipeline.lang_code)
    pack = torch.load(f, weights_only=True)
    pipeline.voices[voice] = pack
    return pack


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )

        self.pipeline = KPipeline(lang_code='a') 
        self.model = self.pipeline.model 

        text = '''
The sky above the port was the color of television, tuned to a dead channel.
"It's not like I'm using," Case heard someone say, as he shouldered his way through the crowd around the door of the Chat. "It's like my body's developed this massive drug deficiency."
It was a Sprawl voice and a Sprawl joke. The Chatsubo was a bar for professional expatriates; you could drink there for a week and never hear two words in Japanese.

These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come. One of the main issues was the shape that Christian churches were to take, since there was not, apparently, a tradition of monumental church buildings when Constantine decided to help the Christian church build a series of truly spectacular structures. The main form that these churches took was that of the basilica, a multipurpose rectangular structure, based ultimately on the earlier Greek stoa, which could be found in most of the great cities of the empire. Christianity, unlike classical polytheism, needed a large interior space for the celebration of its religious services, and the basilica aptly filled that need. We naturally do not know the degree to which the emperor was involved in the design of new churches, but it is tempting to connect this with the secular basilica that Constantine completed in the Roman forum (the so-called Basilica of Maxentius) and the one he probably built in Trier, in connection with his residence in the city at a time when he was still caesar.

[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
'''     

        pack = load_single_voice(self.pipeline, "af_heart").to(self.device)
        text = re.split(r'\n+', text.strip())

        for graphemes in text:
            _, tokens = self.pipeline.g2p(graphemes)
            for gs, ps in self.pipeline.en_tokenize(tokens):
                if not ps:
                    continue
                elif len(ps) > 510:
                    print(f"TODO: Unexpected len(ps) == {len(ps)} > 510 and ps == '{ps}'")
                    continue
                input_ids = self.pipeline.p2ii(ps)
                self.example_inputs = ((input_ids, pack[len(input_ids)-1], 1.0), {})
                break
        assert self.example_inputs is not None


    def get_module(self):
        return self.model, self.example_inputs

    def eval(self) -> Tuple[torch.Tensor]:
        out = self.model(*self.example_inputs[0], **self.example_inputs[1])
        return (out,)

    def train(self):
        raise NotImplementedError("MAML model doesn't support train.")
