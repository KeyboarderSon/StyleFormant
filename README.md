```
python3 synthesize.py --text "The route chosen from the airport to Main Street was the normal one, except where Harwood Street was selected as the means of access to Main Street" --ref_audio RefAudio/lj_02_gt.wav --restore_step 60000 --mode single -p config/LibriTTS/preprocess.yaml -m config/LibriTTS/model.yaml -t config/LibriTTS/train.yaml
```