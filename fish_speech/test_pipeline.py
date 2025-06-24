from fish_speech.lib import Pipeline


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

    for n in range(10):
        print(f"Iteration {n+1}")
        # Example text to synthesize

        text = "Hello, this is a test of the Fish Speech TTS pipeline."
        audio = pipe.inference_engine.inference(text)


if __name__ == "__main__":
    main()