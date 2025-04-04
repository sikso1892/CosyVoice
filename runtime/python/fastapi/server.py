# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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
import logging
from typing import Optional
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import random
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
from funasr import AutoModel

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


initial_seed = random.randint(1, 100000000)

app.state.SEED = initial_seed
set_all_random_seed(app.state.SEED)

def generate_seed():
    return random.randint(1, 100000000)

def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio

def prompt_wav_recognition(prompt_wav):
    res = asr_model.generate(
            input=prompt_wav, 
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
        )
    text = res[0]["text"].split('|>')[-1]
    return text

@app.get("/seed")
async def current_seed():
    return app.state.SEED

@app.put("/seed")
async def change_seed(seed: int):
    set_all_random_seed(seed)
    return await current_seed()

@app.get("/speakers")
async def get_list_available_spks():
    return cosyvoice.list_available_spks()

@app.get("/inference_sft")
async def inference_sft(
    tts_text: str = Form(), 
    spk_id: str = Form(),
    seed: Optional[bool] = Form(False)
):
    if seed:
        new_seed = generate_seed()
        await change_seed(new_seed)
        
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(), 
    prompt_wav: UploadFile = File(),
    prompt_text: Optional[str] = Form(None), 
    seed: Optional[bool] = Form(False)
):
    if not seed:
        new_seed = generate_seed()
        await change_seed(new_seed)
        
        
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    
    if not prompt_text:
        prompt_text = prompt_wav_recognition(prompt_wav.file)
        
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
async def inference_cross_lingual(
    tts_text: str = Form(), 
    prompt_wav: UploadFile = File(),
    seed: Optional[bool] = Form(False)
):
    if not seed:
        new_seed = generate_seed()
        await change_seed(new_seed)
    
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
async def inference_instruct(
    tts_text: str = Form(), 
    spk_id: str = Form(), 
    instruct_text: str = Form(),
    seed: Optional[bool] = Form(False)
):
    if not seed:
        new_seed = generate_seed()
        await change_seed(new_seed)
    
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))

@app.get("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Form(), 
    instruct_text: str = Form(), 
    prompt_wav: UploadFile = File(),
    seed: Optional[bool] = Form(False)
):
    if seed:
        new_seed = generate_seed()
        await change_seed(new_seed)
            
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    
    asr_model_dir = "pretrained_models/SenseVoiceSmall"
    asr_model = AutoModel(
        model=asr_model_dir,
        disable_update=True,
        log_level='DEBUG',
        device="cuda:0"
    )    
        
    
    try:
        cosyvoice = CosyVoice(args.model_dir)

    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            raise TypeError('no valid model_type!')
        
    uvicorn.run(app, host="0.0.0.0", port=args.port)