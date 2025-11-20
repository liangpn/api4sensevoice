from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, UploadFile, File, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from funasr import AutoModel
import numpy as np
import soundfile as sf
import argparse
import uvicorn
from urllib.parse import parse_qs
import os
import re
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from loguru import logger
import sys
import json
import traceback
import time

logger.remove()
log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level}] {file}:{line} - {message}"
logger.add(sys.stdout, format=log_format, level="DEBUG", filter=lambda record: record["level"].no < 40)
logger.add(sys.stderr, format=log_format, level="ERROR", filter=lambda record: record["level"].no >= 40)


class Config(BaseSettings):
    sv_thr: float = Field(0.3, description="Speaker verification threshold")
    chunk_size_ms: int = Field(300, description="Chunk size in milliseconds")
    sample_rate: int = Field(16000, description="Sample rate in Hz")
    bit_depth: int = Field(16, description="Bit depth")
    channels: int = Field(1, description="Number of audio channels")
    avg_logprob_thr: float = Field(-0.25, description="average logprob threshold")

config = Config()

emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "❓",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
	"<|Cry|>": "😭",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "😷",
	"<|Sing|>": "",
	"<|Speech_Noise|>": "",
	"<|withitn|>": "",
	"<|woitn|>": "",
	"<|GBG|>": "",
	"<|Event_UNK|>": "",
}

lang_dict =  {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}

# windows 模型路径
sv_model_path = "D:/ai/models/speech_eres2net_large_sv_zh-cn_3dspeaker_16k"
asr_model_path = "D:/ai/models/SenseVoiceSmall"
asr_offline_model_path = "D:/ai/models/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"
asr_online_model_path = "D:/ai/models/SenseVoiceSmall"
vad_model_path = "D:/ai/models/speech_fsmn_vad_zh-cn-16k-common-pytorch"

# Mac|Linux 模型路径
# sv_model_path = "/Users/liangpn/models/speech_eres2net_large_sv_zh-cn_3dspeaker_16k"
# asr_model_path = "/Users/liangpn/models/SenseVoiceSmall"
# asr_offline_model_path = "/Users/liangpn/models/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"
# asr_online_model_path = "/Users/liangpn/models/SenseVoiceSmall"
# vad_model_path = "/Users/liangpn/models/speech_fsmn_vad_zh-cn-16k-common-pytorch"


def format_str(s):
	for sptk in emoji_dict:
		s = s.replace(sptk, emoji_dict[sptk])
	return s


def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + s
	s = s + emo_dict[emo]

	for emoji in emo_set.union(event_set):
		s = s.replace(" " + emoji, emoji)
		s = s.replace(emoji + " ", emoji)
	return s.strip()

def format_str_v3(s):
	def get_emo(s):
		return s[-1] if s[-1] in emo_set else None
	def get_event(s):
		return s[0] if s[0] in event_set else None

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	for lang in lang_dict:
		s = s.replace(lang, "<|lang|>")
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	new_s = " " + s_list[0]
	cur_ent_event = get_event(new_s)
	for i in range(1, len(s_list)):
		if len(s_list[i]) == 0:
			continue
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			s_list[i] = s_list[i][1:]
		#else:
		cur_ent_event = get_event(s_list[i])
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			new_s = new_s[:-1]
		new_s += s_list[i].strip().lstrip()
	new_s = new_s.replace("The.", " ")
	return new_s.strip()

def contains_chinese_english_number(s: str) -> bool:
    # Check if the string contains any Chinese character, English letter, or Arabic number
    return bool(re.search(r'[\u4e00-\u9fffA-Za-z0-9]', s))


sv_pipeline = pipeline(
    task='speaker-verification',
    model=sv_model_path,
    model_revision='v1.0.0'
)

asr_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model=asr_model_path,
    model_revision="master",
    # device="cuda:0", # 默认使用gpu, cuda:0 指定gpu 、cpu mac或者没有gpu使用
    disable_update=True
)

# 离线 ASR 模型（精确结果）- 支持热词的 Contextual-Paraformer
model_asr_offline = AutoModel(
    model=asr_offline_model_path,
    # device="cuda:0",
    disable_update=True
)

# 在线 ASR 模型（快速结果）- 使用 SenseVoice 保持速度
model_asr_online = AutoModel(
    model=asr_online_model_path,
    trust_remote_code=True,
    remote_code="./model.py",    
    # device="cuda:0",
    disable_update=True
)

model_vad = AutoModel(
    model=vad_model_path,
    model_revision="v2.0.4",
    disable_pbar = True,
    max_end_silence_time=500,
    # speech_noise_thres=0.6,
    disable_update=True,
)

def get_speaker_files():
    """动态获取speaker目录下的所有音频文件"""
    speaker_dir = "speaker"
    if not os.path.exists(speaker_dir):
        return []
    
    audio_extensions = ['.wav']
    files = []
    for file in os.listdir(speaker_dir):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            files.append(os.path.join(speaker_dir, file))
    return files

def reg_spk_init(files):
    reg_spk = {}
    for f in files:
        try:
            data, sr = sf.read(f, dtype="float32")
            k, _ = os.path.splitext(os.path.basename(f))
            reg_spk[k] = {
                "data": data,
                "sr":   sr,
            }
            logger.info(f"加载说话人音频文件: {f}")
        except Exception as e:
            logger.error(f"无法加载音频文件 {f}: {e}")
    return reg_spk

# 动态加载speaker目录下的音频文件
reg_spks_files = get_speaker_files()
reg_spks = reg_spk_init(reg_spks_files)

def speaker_verify(audio, sv_thr, selected_speakers=None):
    """
    说话人验证
    :param audio: 音频数据
    :param sv_thr: 验证阈值
    :param selected_speakers: 选择的说话人列表，如果为None则验证所有说话人
    :return: (是否匹配, 匹配的说话人名称)
    """
    hit = False
    matched_speaker = None
    
    # 如果没有指定说话人，验证所有说话人
    speakers_to_verify = selected_speakers if selected_speakers else list(reg_spks.keys())
    
    for k in speakers_to_verify:
        if k not in reg_spks:
            continue
            
        v = reg_spks[k]
        res_sv = sv_pipeline([audio, v["data"]], sv_thr)
        if res_sv["score"] >= sv_thr:
           hit = True
           matched_speaker = k
        logger.info(f"[speaker_verify] audio_len: {len(audio)}; sv_thr: {sv_thr}; speaker: {k}; score: {res_sv['score']:.3f}; hit: {hit}")
        
        # 如果找到匹配的说话人，可以选择立即返回或继续验证其他说话人
        if hit:
            break
    
    return hit, matched_speaker


def asr_online(audio, lang, cache, use_itn=False, hotwords=None):
    """在线 ASR - 快速结果"""
    start_time = time.time()
    # SenseVoice 不支持热词，所以在线模式不使用热词
    result = model_asr_online.generate(
        input           = audio,
        cache           = cache,
        language        = lang.strip(),
        use_itn         = use_itn,
        batch_size_s    = 60,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug(f"online asr elapsed: {elapsed_time * 1000:.2f} milliseconds")
    return result

def asr_offline(audio, lang, cache, use_itn=False, hotwords=None):
    """离线 ASR - 精确结果，支持热词"""
    start_time = time.time()
    
    # 构建参数
    params = {
        'input': audio,
        'cache': cache,
        'language': lang.strip(),
        'use_itn': use_itn,
        'batch_size_s': 60,
    }
    
    # 如果有热词，添加到参数中
    if hotwords and hotwords.strip():
        params['hotword'] = hotwords.strip()
        logger.info(f"Using hotwords: '{hotwords.strip()}'")
    
    result = model_asr_offline.generate(**params)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug(f"offline asr elapsed: {elapsed_time * 1000:.2f} milliseconds")
    return result

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error("Exception occurred", exc_info=True)
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        message = exc.detail
        data = ""
    elif isinstance(exc, RequestValidationError):
        status_code = HTTP_422_UNPROCESSABLE_ENTITY
        message = "Validation error: " + str(exc.errors())
        data = ""
    else:
        status_code = 500
        message = "Internal server error: " + str(exc)
        data = ""

    return JSONResponse(
        status_code=status_code,
        content=TranscriptionResponse(
            code=status_code,
            msg=message,
            data=data
        ).model_dump()
    )

# Define the response model
class TranscriptionResponse(BaseModel):
    code: int
    info: str
    data: str

class UploadResponse(BaseModel):
    success: bool
    message: str
    filePath: str = ""

class SpeakerListResponse(BaseModel):
    success: bool
    speakers: list[str] = []

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    try:
        query_params = parse_qs(websocket.scope['query_string'].decode())
        sv = query_params.get('sv', ['false'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        lang = query_params.get('lang', ['auto'])[0].lower()
        mode = query_params.get('mode', ['offline'])[0].lower()  # 新增：获取模式参数
        hotwords = query_params.get('hotwords', [''])[0]  # 新增：获取热词参数
        selected_speakers = query_params.get('speakers', [])
        if selected_speakers and selected_speakers[0]:
            selected_speakers = selected_speakers[0].split(',')
        else:
            selected_speakers = None
        
        logger.info(f"WebSocket connection - sv: {sv}, lang: {lang}, mode: {mode}, hotwords: '{hotwords}', speakers: {selected_speakers}")
        
        await websocket.accept()
        chunk_size = int(config.chunk_size_ms * config.sample_rate / 1000)
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)

        cache = {}
        cache_asr_online = {}  # 在线 ASR 缓存
        cache_asr_offline = {}  # 离线 ASR 缓存
        last_vad_beg = last_vad_end = -1
        offset = 0
        hit = False
        
        buffer = b""
        while True:
            data = await websocket.receive_bytes()
            # logger.info(f"received {len(data)} bytes")

            
            buffer += data
            if len(buffer) < 2:
                continue
                
            audio_buffer = np.append(
                audio_buffer, 
                np.frombuffer(buffer[:len(buffer) - (len(buffer) % 2)], dtype=np.int16).astype(np.float32) / 32767.0
            )
            
            # with open('buffer.pcm', 'ab') as f:
            #     logger.debug(f'write {f.write(buffer[:len(buffer) - (len(buffer) % 2)])} bytes to `buffer.pcm`')
                
            buffer = buffer[len(buffer) - (len(buffer) % 2):]
   
            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]
                audio_vad = np.append(audio_vad, chunk)
                
                # with open('chunk.pcm', 'ab') as f:
                #     logger.debug(f'write {f.write(chunk)} bytes to `chunk.pcm`')
                    
                if last_vad_beg > 1:
                    if sv:
                        # speaker verify
                        # If no hit is detected, continue accumulating audio data and check again until a hit is detected
                        # `hit` will reset after `asr`.
                        if not hit:
                            hit, speaker = speaker_verify(audio_vad[int((last_vad_beg - offset) * config.sample_rate / 1000):], config.sv_thr, selected_speakers)
                            if hit:
                                response = TranscriptionResponse(
                                    code=2,
                                    info="detect speaker",
                                    data=speaker
                                )
                                await websocket.send_json(response.model_dump())
                    else:
                        response = TranscriptionResponse(
                            code=2,
                            info="detect speech",
                            data=''
                        )
                        await websocket.send_json(response.model_dump())

                res = model_vad.generate(input=chunk, cache=cache, is_final=False, chunk_size=config.chunk_size_ms)
                # logger.info(f"vad inference: {res}")
                if len(res[0]["value"]):
                    vad_segments = res[0]["value"]
                    for segment in vad_segments:
                        if segment[0] > -1: # speech begin
                            last_vad_beg = segment[0]                           
                        if segment[1] > -1: # speech end
                            last_vad_end = segment[1]
                        if last_vad_beg > -1 and last_vad_end > -1:
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * config.sample_rate / 1000)
                            end = int(last_vad_end * config.sample_rate / 1000)
                            audio_segment = audio_vad[beg:end]
                            logger.info(f"[vad segment] audio_len: {end - beg}")
                            
                            # 说话人验证（如果启用）
                            should_process_asr = True
                            if sv and not hit:
                                logger.info(f"Speaker verification FAILED, skipping ASR for this segment")
                                should_process_asr = False
                            elif sv and hit:
                                logger.info(f"Speaker verification PASSED, processing ASR")
                            
                            # 只有验证通过的音频才进行 ASR
                            if should_process_asr:
                                if mode == "2pass":
                                    # 2pass 模式：先发送在线结果，再发送离线结果
                                    
                                    # 1. 在线 ASR（快速结果）
                                    online_result = asr_online(audio_segment, lang.strip(), cache_asr_online, False, hotwords)
                                    logger.info(f"online asr response: {online_result}")
                                    
                                    if online_result and online_result[0]['text'].strip():
                                        response = TranscriptionResponse(
                                            code=0,
                                            info=json.dumps({
                                                **online_result[0],
                                                "mode": "2pass-online",
                                                "is_final": False
                                            }, ensure_ascii=False),
                                            data=format_str_v3(online_result[0]['text'])
                                        )
                                        await websocket.send_json(response.model_dump())
                                    
                                    # 2. 离线 ASR（精确结果，支持热词）
                                    offline_result = asr_offline(audio_segment, lang.strip(), cache_asr_offline, True, hotwords)
                                    logger.info(f"offline asr response: {offline_result}")
                                    
                                    if offline_result and offline_result[0]['text'].strip():
                                        response = TranscriptionResponse(
                                            code=0,
                                            info=json.dumps({
                                                **offline_result[0],
                                                "mode": "2pass-offline",
                                                "is_final": True
                                            }, ensure_ascii=False),
                                            data=format_str_v3(offline_result[0]['text'])
                                        )
                                        await websocket.send_json(response.model_dump())
                                        
                                elif mode == "online":
                                    # 仅在线模式
                                    result = asr_online(audio_segment, lang.strip(), cache_asr_online, False, hotwords)
                                    logger.info(f"online asr response: {result}")
                                    
                                    if result and result[0]['text'].strip():
                                        response = TranscriptionResponse(
                                            code=0,
                                            info=json.dumps({
                                                **result[0],
                                                "mode": "online",
                                                "is_final": True
                                            }, ensure_ascii=False),
                                            data=format_str_v3(result[0]['text'])
                                        )
                                        await websocket.send_json(response.model_dump())
                                        
                                else:  # offline 模式（默认）
                                    # 仅离线模式，支持热词
                                    result = asr_offline(audio_segment, lang.strip(), cache_asr_offline, True, hotwords)
                                    logger.info(f"offline asr response: {result}")
                                    
                                    if result and result[0]['text'].strip():
                                        response = TranscriptionResponse(
                                            code=0,
                                            info=json.dumps({
                                                **result[0],
                                                "mode": "offline",
                                                "is_final": True
                                            }, ensure_ascii=False),
                                            data=format_str_v3(result[0]['text'])
                                        )
                                        await websocket.send_json(response.model_dump())
                            
                            # 清理状态
                            audio_vad = audio_vad[end:]
                            last_vad_beg = last_vad_end = -1
                            hit = False
                                
                        # logger.debug(f'last_vad_beg: {last_vad_beg}; last_vad_end: {last_vad_end} len(audio_vad): {len(audio_vad)}')

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}\nCall stack:\n{traceback.format_exc()}")
        await websocket.close()
    finally:
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)
        cache.clear()
        logger.info("Cleaned up resources after WebSocket disconnect")


@app.post("/upload-audio", response_model=UploadResponse)
async def upload_audio(audio: UploadFile = File(...), fileName: str = Form(...)):
    try:
        # 确保speaker目录存在
        speaker_dir = "speaker"
        if not os.path.exists(speaker_dir):
            os.makedirs(speaker_dir)
        
        # 构建文件路径
        file_path = os.path.join(speaker_dir, f"{fileName}.wav")
        
        # 保存上传的文件
        with open(file_path, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        logger.info(f"音频文件已保存: {file_path}")
        
        return UploadResponse(
            success=True,
            message="音频上传成功",
            filePath=file_path
        )
        
    except Exception as e:
        logger.error(f"音频上传失败: {e}")
        return UploadResponse(
            success=False,
            message=f"上传失败: {str(e)}"
        )


@app.get("/speakers", response_model=SpeakerListResponse)
async def get_speakers():
    """获取可用的说话人列表"""
    try:
        # 重新加载说话人文件
        global reg_spks, reg_spks_files
        reg_spks_files = get_speaker_files()
        reg_spks = reg_spk_init(reg_spks_files)
        
        speakers = list(reg_spks.keys())
        return SpeakerListResponse(
            success=True,
            speakers=speakers
        )
    except Exception as e:
        logger.error(f"获取说话人列表失败: {e}")
        return SpeakerListResponse(
            success=False,
            speakers=[]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=27000, help='Port number to run the FastAPI app on.')
    parser.add_argument('--certfile', type=str, help='SSL certificate file')
    parser.add_argument('--keyfile', type=str, help='SSL key file')
    args = parser.parse_args()
    
    if args.certfile and args.keyfile:
        uvicorn.run(app, host="0.0.0.0", port=args.port, ssl_certfile=args.certfile, ssl_keyfile=args.keyfile)
    else:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
