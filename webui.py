# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

from funasr import AutoModel

inference_mode_list = ['Pre-trained Voice', '3s Fast Clonning', 'Cross-language Clonning', 'Natural Language Control']
instruct_dict = {
    'Pre-trained Voice': '1. Choose a pre-trained voice\n2. Click the generate audio button',
    '3s Fast Clonning': '1. Select a prompt audio file or record prompt audio, make sure it’s no more than 30s; if both are provided, the prompt audio file is preferred\n2. Enter prompt text\n3. Click the generate audio button',
    'Cross-language Clonning': '1. Select a prompt audio file or record prompt audio, make sure it’s no more than 30s; if both are provided, the prompt audio file is preferred\n2. Click the generate audio button',
    'Natural Language Control': '1. Choose a pre-trained voice\n2. Enter instruction text\n3. Click the generate audio button'
}
stream_mode_list = [('No', False), ('Yes', True)]
max_val = 0.8

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

def prompt_wav_recognition(prompt_wav):
    if prompt_wav:
        res = asr_model.generate(input=prompt_wav,
                                language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                                use_itn=True,
        )
        text = res[0]["text"].split('|>')[-1]
        return text
    
def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['Natural Language Control']:
        if cosyvoice.instruct is False:
            gr.Warning('You are using Natural Language Control mode, the {} model does not support this mode, please use the iic/CosyVoice-300M-Instruct model'.format(args.model_dir))
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text == '':
            gr.Warning('You are using Natural Language Control mode, please enter instruction text')
            yield (cosyvoice.sample_rate, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('You are using Natural Language Control mode, prompt audio/prompt text will be ignored')
    
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different languages
    if mode_checkbox_group in ['Cross-language Clonning']:
        if cosyvoice.instruct is True:
            gr.Warning('You are using Cross-language Clonning mode, the {} model does not support this mode, please use the iic/CosyVoice-300M model'.format(args.model_dir))
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('You are using Cross-language Clonning mode, instruction text will be ignored')
        if prompt_wav is None:
            gr.Warning('You are using Cross-language Clonning mode, please provide prompt audio')
            yield (cosyvoice.sample_rate, default_data)
        gr.Info('You are using Cross-language Clonning mode, please ensure synthesis text and prompt text are in different languages')
    
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s Fast Clonning', 'Cross-language Clonning']:
        if prompt_wav is None:
            gr.Warning('Prompt audio is empty, did you forget to input prompt audio?')
            yield (cosyvoice.sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('Prompt audio sample rate {} is less than {}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (cosyvoice.sample_rate, default_data)
    
    # sft mode only uses sft_dropdown
    if mode_checkbox_group in ['Pre-trained Voice']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('You are using Pre-trained Voice mode, prompt text/prompt audio/instruction text will be ignored!')
        if sft_dropdown == '':
            gr.Warning('No available pre-trained voices!')
            yield (cosyvoice.sample_rate, default_data)
    
    # zero_shot mode only uses prompt_wav prompt text
    if mode_checkbox_group in ['3s Fast Clonning']:
        if prompt_text == '':
            gr.Warning('Prompt text is empty, did you forget to input prompt text?')
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('You are using 3s Fast Clonning mode, pre-trained voice/instruction text will be ignored!')
    
    if mode_checkbox_group == 'Pre-trained Voice':
        logging.info('Received sft inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '3s Fast Clonning':
        logging.info('Received zero_shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == 'Cross-language Clonning':
        logging.info('Received cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('Received instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())

def main():
    with gr.Blocks(title="TTS Archive") as demo:
        gr.Markdown("# TTS Archive")
        
        with gr.Accordion("info. Select the pre-trained model below and follow the instructions to proceed."):
            gr.Markdown(value=
                """                        
                **Used Model:**
                - [CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B)
                - [SenseVoiceSmall-ASR](https://www.modelscope.cn/models/iic/SenseVoiceSmall)
                
                Input the text you want to synthesize, select an inference mode, and follow the instructions."""
            )
        
        
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='Select Inference Mode', value=inference_mode_list[0])            
            with gr.Accordion("instruction"):
                instruction_text = gr.Markdown(label="Instructions", value=instruct_dict[inference_mode_list[0]])
        
        with gr.Row():
            
            sft_dropdown =gr.Dropdown(
                choices=sft_spk, label='Select Pre-trained Voice', value=sft_spk[0] if len(sft_spk) != 0 else '', scale=0.25, 
                visible=(mode_checkbox_group.value in ["Pre-trained Voice", "Natural Language Control"])
            )

            stream = gr.Radio(choices=stream_mode_list, label='Streaming Inference', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="Speed Adjustment (Only supported for non-streaming)", minimum=0.5, maximum=2.0, step=0.1)
        
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="🎲")
                seed = gr.Number(value=0, label="Random Inference Seed")
        
        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Select Prompt Audio File, Sampling Rate Not Less Than 16kHz', visible=("Clonning" in mode_checkbox_group.value))
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record Prompt Audio', visible=("Clonning" in mode_checkbox_group.value))
            prompt_text = gr.Textbox(label="Prompt Transcription", lines=3, placeholder="Prompt transcription (auto ASR, you can correct the recognition results)", value='', visible=("Clonning" in mode_checkbox_group.value))
            tts_text = gr.Textbox(label="Input Text for Synthesis", lines=3, value="새해 복 많이 받으세요. Happy New Year. 新年明けましておめでとうございます 新年快乐")            
            instruct_text = gr.Textbox(label="Input Instruct Text", lines=3, placeholder="Please enter instruct text.", value='')
        
        generate_button = gr.Button("Generate Audio")
        audio_output = gr.Audio(label="Synthesized Audio", autoplay=True, streaming=True)
        
        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
        mode_checkbox_group.change(
            fn=lambda mode: (
                gr.update(visible=(mode in ['Pre-trained Voice', 'Natural Language Control'])),
                gr.update(visible=("Clonning" in mode)),
                gr.update(visible=("Clonning" in mode)),
                gr.update(visible=("Clonning" in mode))
            ), 
            inputs=[mode_checkbox_group], 
            outputs=[sft_dropdown, prompt_wav_upload, prompt_wav_record, prompt_text]
        )
        
        prompt_wav_upload.change(fn=prompt_wav_recognition, inputs=[prompt_wav_upload], outputs=[prompt_text])
        prompt_wav_record.change(fn=prompt_wav_recognition, inputs=[prompt_wav_record], outputs=[prompt_text])        

        demo.queue(max_size=4, default_concurrency_limit=2)
        demo.launch(server_name='0.0.0.0', server_port=args.port, favicon_path="./asset/favicon.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=os.getenv("PORT", 27860))
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    
    cosyvoice = CosyVoice2(args.model_dir) if 'CosyVoice2' in args.model_dir else CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_available_spks()
    prompt_sr = 16_000
    default_data = np.zeros(cosyvoice.sample_rate)
    model_dir = "pretrained_models/SenseVoiceSmall"
    asr_model = AutoModel(
        model=model_dir,
        disable_update=True,
        log_level='DEBUG',
        device="cuda:0")
        
    main()