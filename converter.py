from pydub import AudioSegment
import json
desc_file='train_corpus.json'

audio_paths, durations, texts = [], [], []
with open(desc_file) as json_line_file:
    for line_num, json_line in enumerate(json_line_file):
        try:
            spec = json.loads(json_line)
            # if float(spec['duration']) > max_duration:
            #     continue
            audio_paths.append(spec['key'])
            # durations.append(float(spec['duration']))
            # texts.append(spec['text'])
        except Exception as e:
            # Change to (KeyError, ValueError) or
            # (KeyError,json.decoder.JSONDecodeError), depending on
            # json module version
            print('Error reading line #{}: {}'
                  .format(line_num, json_line))

# print (audio_paths)
audio_paths = ['..'+path[:-3]+'flac' for path in audio_paths]

for input_file in audio_paths:
    print (input_file)
    output_file = input_file[:-4]+'wav'
    flac_audio = AudioSegment.from_file(input_file, format="flac")
    flac_audio.export(output_file, format="wav")



