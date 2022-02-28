# StyleFormant

KeonLee님이 구현하신 StyleSpeech와 FastPitchFormant의 구성 요소를 합쳤습니다. 특별히 StyleSpeech의 few shot을 target으로 한 task1과, 텍스트와 운율을 별도로 모델링하는 task2를 복합적으로 모델링 하였습니다. 

ksc2021에서 학부생 포스터 발표를 진행하였고 이에 해당하는 [paper](http://ksc2021.kiise.or.kr/wp/popPDF.asp?p=jncAE2qcBEA1CDPjT0QTWer1lVVO218WOJw3Yxr0W0I4mwa0QmlOw1v5gXc2) 입니다. 


## Usage

### Pretrained Model
* [StyleFormant](https://drive.google.com/file/d/1MxhvvqXQuF8vCyF92cDYOJ0wM5HPTy0R/view?usp=sharing)
* [StyleFormant+Meta](https://drive.google.com/file/d/1J-8TvID-Q-ezNBZZI9rrqO02MqGLJrAB/view?usp=sharing)

### Quick Inference
1) pretrained model을 따로 폴더의 경로를 만들어,  ```./output/ckpt/LibriTTS_meta_learner``` 에 위치하도록 해주세요.


2) 
```
python synthesize.py --text "TEXT" --ref_audio RefAudio/lj_02_gt.wav --restore_step 60000 --mode single -p config/LibriTTS/preprocess.yaml -m config/LibriTTS/model.yaml -t config/LibriTTS/train.yaml
```
를 실행하면 ```./output/result/LibriTTS_meta_learner``` 에 실행 결과가 나타납니다.

### Preprocess
#### Quick Start
keonlee님이 제공해주신 [StyleSpeech](https://github.com/keonlee9420/StyleSpeech)의 TextGrid들 중 LibriTTS.zip을 다운로드하고 unzip한 폴더를  ```preprocessed_data/LibriTTS/TextGrid/``` 하단에 위치시킵니다.

#### Start with scratch

1) [Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases?page=3
)를 다운로드합니다.

2) LibriTTS 데이터셋을 다운받아 raw_data의 하위 폴더로 넣어줍니다.

3) 둘 중 하나의 명령어로 alignment 작업을 진행합니다.
```
./montreal-forced-aligner/bin/mfa_align raw_data/LibriTTS/ lexicon/librispeech-lexicon.txt english preprocessed_data/LibriTTS
```  
혹은 
```
./montreal-forced-aligner/bin/mfa_train_and_align raw_data/LibriTTS/ lexicon/librispeech-lexicon.txt preprocessed_data/LibriTTS
```

### Train
```
python train.py -p config/LibriTTS/preprocess.yaml -m config/LibriTTS/model.yaml -t config/LibriTTS/train.yaml
```

### Inference
```
python3 synthesize.py --text "Hello world." --ref_audio RefAudio/lj_02_gt.wav --restore_step 60000 --mode single -p config/LibriTTS/preprocess.yaml -m config/LibriTTS/model.yaml -t config/LibriTTS/train.yaml
```


## Reference
https://github.com/keonlee9420/StyleSpeech
https://github.com/KevinMIN95/StyleSpeech (official)
https://github.com/keonlee9420/FastPitchFormant