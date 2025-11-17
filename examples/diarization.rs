/*
Download the Sortformer model:
https://huggingface.co/altunenes/parakeet-rs/blob/main/diar_sortformer_4spk-v1.onnx

Download test audio:
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav

Usage:
cargo run --example diarization --features sortformer 6_speakers.wav

NOTE: This example combines two NVIDIA models:
- Parakeet-TDT: Provides transcription with sentence-level timestamps
- Sortformer: Provides speaker identification (4 speakers max)
- We use TDT's sentence timestamps + Sortformer's speaker IDs
- Even if Sortformer can't detect a segment, we still get the transcription (marked UNKNOWN)

MEMORY LIMITATION (Sortformer v1 Model):
This model has significant memory requirements that increase with audio duration.
According to NVIDIA's official documentation:
  "The maximum duration of a test recording depends on available GPU memory.
   For an RTX A6000 48GB model, the limit is around 12 minutes."

Expected memory usage (based on my tests):
  • 5 minutes:  ~8-10 GB RAM
  • 10 minutes: ~15-17 GB RAM
  • 15+ minutes: Will likely crash on systems with <32GB RAM

This is a known limitation of the Sortformer v1 model architecture.
For long audio files, consider:
  - Processing shorter segments separately

NOTE: We could reduce RAM usage by increasing HOP_LENGTH in src/sortformer.rs (e.g., 160→320),
which would cut memory in half. However, this breaks the model's expected input format since
it was trained with specific NeMo preprocessing parameters. The quality degrades and we'd still
hit the model's inherent sequence length limits. Rather than fighting the model's architecture
with workarounds, it's better to wait for NVIDIA's next generation models with improved efficiency.

*/

#[cfg(feature = "sortformer")]
use parakeet_rs::sortformer::Sortformer;
#[cfg(feature = "sortformer")]
use parakeet_rs::TimestampMode;
#[cfg(feature = "sortformer")]
use hound;
#[cfg(feature = "sortformer")]
use std::env;
#[cfg(feature = "sortformer")]
use std::time::Instant;

#[allow(unreachable_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "sortformer"))]
    {
        eprintln!("Error: This example requires the 'sortformer' feature.");
        eprintln!("Please run with: cargo run --example diarization --features sortformer <audio.wav>");
        return Err("sortformer feature not enabled".into());
    }

    #[cfg(feature = "sortformer")]
    {
        let start_time = Instant::now();
        let args: Vec<String> = env::args().collect();
        let audio_path = args.get(1)
            .expect("Please specify audio file: cargo run --example diarization --features sortformer <audio.wav>");

        println!("{}", "=".repeat(80));
        println!("Step 1/3: Loading audio...");

        let mut reader = hound::WavReader::open(audio_path)?;
        let spec = reader.spec();

        let audio: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .collect::<Result<Vec<_>, _>>()?,
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.map(|s| s as f32 / 32768.0))
                .collect::<Result<Vec<_>, _>>()?,
        };

        println!("Loaded {} samples ({} Hz, {} channels)", audio.len(), spec.sample_rate, spec.channels);

        println!("{}", "=".repeat(80));
        println!("Step 2/3: Performing speaker diarization with NVIDIA Sortformer...");

        // Perform diarization
        let mut sortformer = Sortformer::new("diar_sortformer_4spk-v1.onnx")?;
        let speaker_segments = sortformer.diarize(audio.clone(), spec.sample_rate, spec.channels)?;

        println!("Found {} speaker segments from Sortformer", speaker_segments.len());

        println!("{}", "=".repeat(80));
        println!("Step 3/3: Transcribing with Parakeet-TDT and attributing speakers...\n");

        // Use TDT for transcription with sentence-level timestamps
        let mut parakeet = parakeet_rs::ParakeetTDT::from_pretrained("./tdt", None)?;

        // Transcribe with Sentences mode (TDT provides punctuation for proper segmentation)
        if let Ok(result) = parakeet.transcribe_samples(audio, spec.sample_rate, spec.channels, Some(TimestampMode::Sentences)) {
            // For each sentence from TDT, find the corresponding speaker from Sortformer
            for segment in &result.tokens {
                // Find which speaker was active during this segment's midpoint
                let segment_mid = (segment.start + segment.end) / 2.0;
                let speaker = speaker_segments
                    .iter()
                    .find(|s| segment_mid >= s.start && segment_mid <= s.end)
                    .map(|s| format!("Speaker {}", s.speaker_id))
                    .unwrap_or_else(|| "UNKNOWN".to_string());

                println!("[{:.2}s - {:.2}s] {}: {}",
                    segment.start, segment.end, speaker, segment.text);
            }
        }

        println!("\n{}", "=".repeat(80));
        let elapsed = start_time.elapsed();
        println!("\n✓ Diarization and transcription completed in {:.2}s", elapsed.as_secs_f32());
        println!("• UNKNOWN: Segments where no speaker was detected by Sortformer");

        Ok(())
    }

    #[cfg(not(feature = "sortformer"))]
    unreachable!()
}
