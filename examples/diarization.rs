/*
Speaker Diarization with NVIDIA Nemotron-3 Diarization (Sortformer v3, up to 8 speakers).

Download nemotron3_diar_v3.onnx from https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-3-diarization
(or export it with scripts/export_diar_sortformer.py) into the working directory, then:
  cargo run --release --example diarization --features sortformer -- audio.wav [model.onnx]

  preset: offline (default) | low | very-low | ultra
  model:  defaults to nemotron3_diar_v3.onnx

Audio: 16 kHz mono WAV. Speaker IDs are ordered by each speaker's first arrival.

Output resolution (10ms vs 80ms):
  The model works at two rates internally. 80ms "diar" frames drive the speaker cache;
  10ms "hires" frames are the actual output. `diarize()` builds segments from the 10ms
  predictions, so boundaries are accurate to ~10ms. Want the raw signal instead of
  segments? `predict_raw(audio, 16000, 1)` returns per 10ms frame [T, 8] probabilities.

Latency vs accuracy (same ONNX, no re-export; set before feeding audio):
  Buffer latency = (chunk_len + right_context) * 80ms. NVIDIA's recommended presets:
      StreamingProfile::offline()            30.4 s  (default, fastest; best for files)
      StreamingProfile::low_latency()        1.04 s  (live audio)
      StreamingProfile::very_low_latency()   0.64 s
      StreamingProfile::ultra_low_latency()  0.32 s  (you might need a gpu for this one)
  Lower latency means more model steps per second of audio, so more compute.
  A custom StreamingProfile { .. } also works. `diarizer.latency()` reports the buffer.
*/

#[cfg(not(feature = "sortformer"))]
fn main() {
    eprintln!("This example requires the 'sortformer' feature: --features sortformer");
}

#[cfg(feature = "sortformer")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::sortformer::{Sortformer, StreamingProfile};
    use std::env;
    use std::time::Instant;

    let args: Vec<String> = env::args().collect();
    let audio_path = args
        .get(1)
        .expect("usage: diarization <audio.wav> [offline|low|very-low|ultra] [model.onnx]");
    let onnx = args.get(3).map(String::as_str).unwrap_or("nemotron3_diar_v3.onnx");
    let profile = match args.get(2).map(String::as_str).unwrap_or("offline") {
        "offline" => StreamingProfile::offline(),
        "low" => StreamingProfile::low_latency(),
        "very-low" => StreamingProfile::very_low_latency(),
        "ultra" => StreamingProfile::ultra_low_latency(),
        other => return Err(format!("unknown preset {other}: offline | low | very-low | ultra").into()),
    };

    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
    };
    let duration = audio.len() as f32 / spec.sample_rate as f32 / spec.channels as f32;
    println!("Loaded {:.1}s ({} Hz, {} ch)", duration, spec.sample_rate, spec.channels);

    let mut diarizer = Sortformer::new(onnx)?;
    diarizer.set_profile(profile)?;
    println!("Latency profile: {:.2}s buffer", diarizer.latency());

    let started = Instant::now();
    let segments = diarizer.diarize(audio, spec.sample_rate, spec.channels)?;
    println!("Diarized in {:.2}s\n", started.elapsed().as_secs_f32());

    for seg in &segments {
        println!(
            "[{:7.2}s - {:7.2}s] speaker_{}",
            seg.start as f64 / 16_000.0,
            seg.end as f64 / 16_000.0,
            seg.speaker_id
        );
    }
    Ok(())
}
