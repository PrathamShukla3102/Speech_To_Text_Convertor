import os
import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import jiwer


def load_audio(file_path):
    speech_array,sampling_rate = torchaudio.load(file_path)
    if sampling_rate != 16000:
        speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(speech_array)
    return speech_array.squeeze().numpy()


def load_text(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def evaluate_cer_wer_scores(audio_dir, text_dir):
    print("Initializing Wav2Vec2 model and processor...")
    model_name = 'facebook/wav2vec2-large-robust-ft-libri-960h'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()
    print("Model initialization complete.")

#variables

    cer_scores = []
    wer_scores = []
    total_samples = 0
    skipped_files = []

    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(audio_dir, audio_file)
            text_file = audio_file.replace('.wav', '.txt')
            text_path = os.path.join(text_dir, text_file)

            if os.path.exists(text_path):
                # Load and preprocess the audio
                input_audio = load_audio(audio_path)
                input_values = processor(input_audio, sampling_rate=16000, return_tensors="pt").input_values

                # Perform ASR
                with torch.no_grad():
                    logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]

                # Load reference transcription
                reference = load_text(text_path)

                # Skip empty references
                if not reference.strip():
                    print(f"Skipping file {audio_file} due to empty reference.")
                    skipped_files.append(audio_file)
                    continue

                try:
                    # Compute CER and WER
                    cer = jiwer.cer(reference, transcription)
                    wer = jiwer.wer(reference, transcription)
                    cer_scores.append(cer)
                    wer_scores.append(wer)
                    total_samples += 1

                    print(f"File: {audio_file}")
                    print(f"Reference: {reference}")
                    print(f"Transcriptions: {transcription}")
                    print("Computing CER and WER...")
                    print(f"CER: {cer:.4f}")
                    print(f"WER: {wer:.4f}")
                except ValueError as e:
                    print(f"Error processing file {audio_file}: {str(e)}")
                    skipped_files.append(audio_file)

    if total_samples > 0:
        average_cer = sum(cer_scores) / total_samples
        average_wer = sum(wer_scores) / total_samples
        print(f"Average CER Value: {average_cer:.4f}")
        print(f"Average WER Value: {average_wer:.4f}")
    else:
        print("No valid samples processed.")

    if skipped_files:
        print(f"Skipped files: {', '.join(skipped_files)}")


audio_directory = '../Indian_accent_Speech_to_text_convertor/dataset/audio'
text_directory = '../Indian_accent_Speech_to_text_convertor/dataset/transcriptions'
evaluate_cer_wer_scores(audio_directory, text_directory)