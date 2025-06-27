from fish_speech.lib import Pipeline
from pathlib import Path
import torch
from fish_speech.utils.schema import Reference
import whisper

def main():
    llama_path = "/home/timo/openaudio_models"
    decoder_path = "/home/timo/openaudio_models/codec.pth"
    decoder_config = "modded_dac_vq"
    device = "cuda"
    half = False
    compile = False

    pipe = Pipeline(llama_path=llama_path,
                    decoder_path=decoder_path,
                    decoder_config=decoder_config,
                    device=device,
                    half=half,
                    compile=compile)

    text = "Hello, this is a test of the Fish Speech TTS pipeline."
    ref_path = Path(__file__).parent / "cce035b9-4764-40ab-b18f-d2d5cfd2e88f.pt"
    ref_text = " Well, the crime has been committed and it so happens that I'm of such an age that I was able to see a beginning. And so I just enjoy saying, doop doop doop, on the contrary. I'm not sure I'll enjoy it. I'm trying to throw all excitement, pleasure, joy, joy, joy. But if you've got any sense of responsibility, you can't do it."
    ref = Reference(tokens=torch.load(ref_path), text=ref_text)

    geert_audio_path = Path(__file__).parent / "Cartman.wav"
    model = whisper.load_model("base", device=device)
    result = model.transcribe(str(geert_audio_path))
    del model
    torch.cuda.empty_cache()
    geert_transcribe_text = result["text"]
    new_ref = pipe.make_reference(str(geert_audio_path), text=geert_transcribe_text)

    for n in range(10):
        print(f"Iteration {n+1}")
        res = pipe.generate(
            text=text,
            references=[ref],
            seed=42,
        )


        



        


if __name__ == "__main__":
    main()