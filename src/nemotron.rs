use crate::decoder::{TimedToken, TranscriptionResult};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_nemotron::{NemotronEncoderCache, NemotronModel};
use crate::timestamps::{process_timestamps, rebuild_text, TimestampMode};
use ndarray::{s, Array2, Array3};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex};

// Nemotron 0.6B model constants
// note that those numbers are coming from offical impl. and of course my onnx export decisions.
// Buffer logic and cache slicing strategy derived from:
// https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/utils/streaming_utils.py
// https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/audio_preprocessing.py
const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;

// Streaming chunk config (identical across English-only and multilingual variants:
// both use chunk_size_output=7 in NeMo's streaming_cfg which corresponds to 56 mel frames).
const CHUNK_SIZE: usize = 56;
const PRE_ENCODE_CACHE: usize = 9;
// FastConformer subsamples mel frames by 8x, so each encoder output frame spans
// SUBSAMPLING_FACTOR * HOP_LENGTH audio samples (80ms at 16kHz). Used to convert
// the encoder frame index of an emitted token into a wall-clock timestamp.
const SUBSAMPLING_FACTOR: usize = 8;

/// Token-level transcription output with its per-token log-probability.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TokenInfo {
    /// Vocabulary id of the emitted (non-blank) token. `usize` to match the
    /// rest of this module (`decode_single`, `accumulated_tokens`, vocab ids).
    pub id: usize,
    /// Decoded SentencePiece piece for this id. For the multilingual variant
    /// language-tag pieces (e.g. `<en-US>`) are returned verbatim, not stripped.
    pub text: String,
    /// Log-softmax value of the chosen token over the full vocab logits.
    pub logprob: f32,
    /// Encoder frame index at which this token was emitted (one frame spans
    /// 80ms = `SUBSAMPLING_FACTOR * HOP_LENGTH / SAMPLE_RATE`). Used to derive
    /// per-token timestamps. `decode_chunk_tokens` populates it chunk-local
    /// (0-based); the offline `process_audio` path then rebases it to a global
    /// index across the whole utterance, so `transcribe_audio_with_tokens` and
    /// `transcribe_audio_with_timestamps` report absolute positions. The
    /// streaming `process_chunk` path leaves it chunk-local.
    pub local_frame: usize,
}

/// Numerically stable log-softmax value at `idx` over the full logits vector.
///
/// `max_logit` is the maximum logit, which the greedy decode loop already
/// computes while taking the argmax — passing it in lets us walk the logits
/// just once more (the exp-sum) instead of a second max pass.
fn log_softmax_at(logits: &[f32], idx: usize, max_logit: f32) -> f32 {
    let lse = max_logit + logits.iter().map(|x| (x - max_logit).exp()).sum::<f32>().ln();
    logits[idx] - lse
}

/// Language → prompt embedding index for the multilingual model. Mirrors
/// `cfg.model_defaults.prompt_dictionary` from the .nemo. Embedded here so
/// we don't require a sidecar `config.json` next to the ONNX files.
///
/// NVIDIA's model card documents 40 language-locales across 3 tiers:
///   - **Transcription-ready (19):** en, es, fr, it, pt, nl, de, tr, ru, ar,
///     hi, ja, ko, vi, uk (with locales).
///   - **Broad-coverage (13):** pl, sv, cs, nb, da, bg, fi, hr, sk, zh-CN,
///     hu, ro, et.
///   - **Adaptation-ready (8):** el, lt, lv, mt, sl, he, th, nn — recognized
///     by the tokenizer but need fine-tuning for production quality.
///
/// The full dictionary below contains additional entries because (a) several
/// codes alias the same prompt index (`en` == `en-US`, `hi` == `hi-IN`, ...)
/// and (b) some experimental languages (e.g. `qu-PE`, `mi-NZ`, `haw-US`)
/// have prompt slots but are not in the model card. Using those will work
/// but quality is not guaranteed.
///
/// See: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
const PROMPT_DICTIONARY: &[(&str, i64)] = &[
    ("af-ZA", 54), ("am-ET", 49), ("ar", 7), ("ar-AR", 7), ("auto", 101),
    ("ay-BO", 81), ("az-AZ", 66), ("bg", 30), ("bg-BG", 30), ("bn-IN", 36),
    ("cs", 22), ("cs-CZ", 22), ("da", 25), ("da-DK", 25), ("de", 9),
    ("de-DE", 9), ("el", 21), ("el-GR", 21), ("en", 0), ("en-GB", 1),
    ("en-US", 0), ("enGB", 1), ("es", 3), ("es-ES", 2), ("es-US", 3),
    ("esES", 2), ("et", 60), ("et-EE", 60), ("fa-IR", 38), ("fi", 26),
    ("fi-FI", 26), ("fr", 8), ("fr-CA", 100), ("fr-FR", 8), ("gn-PY", 82),
    ("gu-IN", 42), ("ha-NG", 50), ("haw-US", 97), ("he-IL", 64), ("hi", 6),
    ("hi-HI", 6), ("hi-IN", 6), ("hr", 29), ("hr-HR", 29), ("hu", 23),
    ("hu-HU", 23), ("hy-AM", 68), ("id-ID", 34), ("ig-NG", 53), ("it", 15),
    ("it-IT", 15), ("ja-JA", 10), ("ja-JP", 10), ("ka-GE", 67), ("km-KH", 47),
    ("kn-IN", 43), ("ko", 14), ("ko-KO", 14), ("ko-KR", 14), ("ku-TR", 65),
    ("ky-KG", 71), ("ln-CD", 58), ("lt", 31), ("lt-LT", 31), ("lv", 61),
    ("lv-LV", 61), ("mi-NZ", 96), ("ml-IN", 44), ("mr-IN", 41), ("ms-MY", 35),
    ("mt-MT", 102), ("nah-MX", 83), ("nb", 103), ("nb-NO", 103), ("ne-NP", 46),
    ("nl", 16), ("nl-NL", 16), ("nn", 104), ("nn-NO", 104), ("no", 27),
    ("no-NO", 27), ("ny-MW", 57), ("or-KE", 59), ("pl", 17), ("pl-PL", 17),
    ("pt", 13), ("pt-BR", 12), ("pt-PT", 13), ("qu-PE", 80), ("ro", 20),
    ("ro-RO", 20), ("ru", 11), ("ru-RU", 11), ("rw-RW", 55), ("si-LK", 45),
    ("sk", 28), ("sk-SK", 28), ("sl", 62), ("sl-SI", 62), ("sm-WS", 98),
    ("so-SO", 56), ("sv", 24), ("sv-SE", 24), ("sw-KE", 48), ("ta-IN", 39),
    ("te-IN", 40), ("tg-TJ", 70), ("th-TH", 32), ("to-TO", 99), ("tr", 18),
    ("tr-TR", 18), ("uk", 19), ("uk-UA", 19), ("ur-PK", 37), ("uz-UZ", 69),
    ("vi-VN", 33), ("yo-NG", 52), ("zh-CN", 4), ("zh-TW", 5), ("zh-ZH", 4),
    ("zu-ZA", 51),
];

/// Detect SentencePiece pieces that encode a language tag like `<en-US>` or
/// `<en>`. The multilingual model emits these inline with text; they're
/// stripped from the user-visible transcript.
fn is_lang_tag(piece: &str) -> bool {
    let bytes = piece.as_bytes();
    if bytes.len() < 4 || bytes[0] != b'<' || bytes[bytes.len() - 1] != b'>' {
        return false;
    }
    let inner = &bytes[1..bytes.len() - 1];
    match inner.len() {
        2 => inner[0].is_ascii_lowercase() && inner[1].is_ascii_lowercase(),
        5 => inner[0].is_ascii_lowercase()
            && inner[1].is_ascii_lowercase()
            && inner[2] == b'-'
            && inner[3].is_ascii_uppercase()
            && inner[4].is_ascii_uppercase(),
        _ => false,
    }
}

/// Which Nemotron variant a handle wraps. Detected automatically from
/// the encoder ONNX graph (multilingual exposes a `prompt_index` input).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NemotronMode {
    /// English-only Nemotron 0.6B (vocab 1024, no language conditioning).
    EnglishOnly,
    /// Multilingual Nemotron 3.5 0.6B with `prompt_index` input,
    /// vocab ~13k, supports `target_lang` selection.
    Multilingual,
}

/// Minimal SentencePiece vocabulary loader.
/// Parses the protobuf .model file to extract token strings.
/// Note that, our vocab.rs cannot parse protobuf format. I haven't test it with digit spacing yet, at least for this initial impl.
pub struct SentencePieceVocab {
    pieces: Vec<String>,
}

impl SentencePieceVocab {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| Error::Tokenizer(format!("Failed to open tokenizer.model: {e}")))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| Error::Tokenizer(format!("Failed to read tokenizer.model: {e}")))?;

        let pieces = Self::parse_sentencepiece_model(&data)?;
        Ok(Self { pieces })
    }

    fn parse_sentencepiece_model(data: &[u8]) -> Result<Vec<String>> {
        let mut pieces = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            let (field_header, bytes_read) = Self::read_varint(&data[pos..])?;
            pos += bytes_read;

            let field_num = field_header >> 3;
            let wire_type = field_header & 0x7;

            match (field_num, wire_type) {
                (1, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;

                    if pos + len as usize > data.len() {
                        break;
                    }

                    let piece_data = &data[pos..pos + len as usize];
                    pos += len as usize;

                    if let Ok(piece) = Self::parse_piece_message(piece_data) {
                        pieces.push(piece);
                    }
                }
                (_, 0) => {
                    let (_, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;
                }
                (_, 1) => pos += 8,
                (_, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read + len as usize;
                }
                (_, 5) => pos += 4,
                _ => break,
            }
        }

        if pieces.is_empty() {
            return Err(Error::Tokenizer("No tokens found in model".into()));
        }

        Ok(pieces)
    }

    fn parse_piece_message(data: &[u8]) -> Result<String> {
        let mut pos = 0;
        let mut piece = String::new();

        while pos < data.len() {
            let (field_header, bytes_read) = Self::read_varint(&data[pos..])?;
            pos += bytes_read;

            let field_num = field_header >> 3;
            let wire_type = field_header & 0x7;

            match (field_num, wire_type) {
                (1, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;

                    if pos + len as usize <= data.len() {
                        piece = String::from_utf8_lossy(&data[pos..pos + len as usize]).to_string();
                    }
                    pos += len as usize;
                }
                (_, 0) => {
                    let (_, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;
                }
                (_, 1) => pos += 8,
                (_, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read + len as usize;
                }
                (_, 5) => pos += 4,
                _ => break,
            }
        }

        Ok(piece)
    }

    fn read_varint(data: &[u8]) -> Result<(u64, usize)> {
        let mut result: u64 = 0;
        let mut shift = 0;
        let mut pos = 0;

        while pos < data.len() && pos < 10 {
            let byte = data[pos];
            result |= ((byte & 0x7F) as u64) << shift;
            pos += 1;

            if byte & 0x80 == 0 {
                return Ok((result, pos));
            }
            shift += 7;
        }

        Err(Error::Tokenizer("Invalid varint".into()))
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut result = String::new();
        for &id in ids {
            if id < self.pieces.len() {
                let piece = &self.pieces[id];
                let decoded = piece.replace('\u{2581}', " ");
                result.push_str(&decoded);
            }
        }
        result.trim_start().to_string()
    }

    pub fn decode_single(&self, id: usize) -> String {
        if id < self.pieces.len() {
            self.pieces[id].replace('\u{2581}', " ")
        } else {
            String::new()
        }
    }

    pub fn size(&self) -> usize {
        self.pieces.len()
    }

    /// Token IDs whose SentencePiece pieces look like language tags
    /// (`<en-US>`, `<fr>`, ...). and ofc empty for the en only vocab.
    pub fn lang_tag_ids(&self) -> Vec<usize> {
        self.pieces
            .iter()
            .enumerate()
            .filter_map(|(i, p)| is_lang_tag(p).then_some(i))
            .collect()
    }
}

/// Shared handle to a loaded Nemotron model.
/// ONNX session is only loaded once and reference counted.
///
/// Use [`NemotronHandle::load`] to load from disk, then [`Nemotron::from_shared`]
/// to spawn each stream with its own independent decoder state.
/// Variant is auto-detected: both the en only 0.6B and the multi lang
/// 3.5 0.6B drop into the same type.
#[derive(Clone)]
pub struct NemotronHandle {
    model: Arc<Mutex<NemotronModel>>,
    vocab: Arc<SentencePieceVocab>,
    mel_basis: Arc<Array2<f32>>,
    mode: NemotronMode,
    num_encoder_layers: usize,
    hidden_dim: usize,
    left_context: usize,
    conv_context: usize,
    decoder_lstm_dim: usize,
    decoder_lstm_layers: usize,
    vocab_size: usize,
    blank_id: usize,
    /// Empty for en. populated for multilingual with all `<xx-XX>` token ids.
    lang_tag_ids: Arc<Vec<usize>>,
}

/// Nemotron streaming ASR model (0.6B parameters).
/// We dont apply mel normalization unlike others...
///
/// For a single stream, use [`Nemotron::from_pretrained`]. For multiple
/// concurrent streams (e.g. mic + system audio) sharing one loaded model,
/// use [`NemotronHandle::load`] followed by [`Nemotron::from_shared`].
///
/// For the multilingual variant call [`Nemotron::set_target_lang`] before
/// transcribing if you know the language; otherwise it defaults to `auto`
/// (prompt index 101) and lets the model pick.
pub struct Nemotron {
    model: Arc<Mutex<NemotronModel>>,
    vocab: Arc<SentencePieceVocab>,
    mel_basis: Arc<Array2<f32>>,
    mode: NemotronMode,
    num_encoder_layers: usize,
    hidden_dim: usize,
    left_context: usize,
    conv_context: usize,
    vocab_size: usize,
    blank_id: usize,
    lang_tag_ids: Arc<Vec<usize>>,
    encoder_cache: NemotronEncoderCache,
    state_1: Array3<f32>,
    state_2: Array3<f32>,
    last_token: i32,
    /// `None` for English-only mode; `Some(idx)` for multilingual.
    prompt_index: Option<i64>,
    /// Raw audio sample buffer for proper mel computation
    audio_buffer: Vec<f32>,
    /// How many audio samples have been processed (converted to mel and sent to encoder)
    audio_processed: usize,
    chunk_idx: usize,
    accumulated_tokens: Vec<usize>,
}

impl NemotronHandle {
    /// Load the Nemotron model and vocabulary from a directory.
    ///
    /// Required files in `path`:
    /// - `encoder.onnx` + `encoder.onnx.data`
    /// - `decoder_joint.onnx`
    /// - `tokenizer.model`
    ///
    /// The returned handle is cheap to clone and can be used to spawn any
    /// number of [`Nemotron`] instances via [`Nemotron::from_shared`], each
    /// with its own independent decoder state.
    pub fn load<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let path = path.as_ref();

        let vocab = SentencePieceVocab::from_file(path.join("tokenizer.model"))?;
        let vocab_size = vocab.size();

        let exec = exec_config.unwrap_or_default();
        let model = NemotronModel::from_pretrained(path, exec, vocab_size)?;
        let mel_basis = crate::audio::create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE);

        let mode = if model.has_prompt {
            NemotronMode::Multilingual
        } else {
            NemotronMode::EnglishOnly
        };
        let cfg = model.config.clone();
        let lang_tag_ids = if mode == NemotronMode::Multilingual {
            vocab.lang_tag_ids()
        } else {
            Vec::new()
        };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            vocab: Arc::new(vocab),
            mel_basis: Arc::new(mel_basis),
            mode,
            num_encoder_layers: cfg.num_encoder_layers,
            hidden_dim: cfg.hidden_dim,
            left_context: cfg.left_context,
            conv_context: cfg.conv_context,
            decoder_lstm_dim: cfg.decoder_lstm_dim,
            decoder_lstm_layers: cfg.decoder_lstm_layers,
            vocab_size: cfg.vocab_size,
            blank_id: cfg.blank_id,
            lang_tag_ids: Arc::new(lang_tag_ids),
        })
    }

    /// Which variant this handle wraps (auto detected at load time).
    pub fn mode(&self) -> NemotronMode {
        self.mode
    }

    /// Languages this model can transcribe, as accepted by
    /// [`Nemotron::set_target_lang`]. Empty for the English-only variant.
    pub fn available_languages(&self) -> Vec<&'static str> {
        match self.mode {
            NemotronMode::Multilingual => PROMPT_DICTIONARY.iter().map(|(k, _)| *k).collect(),
            NemotronMode::EnglishOnly => Vec::new(),
        }
    }
}

impl Nemotron {
    /// Load Nemotron from a directory and return a ready to use instance.
    /// Convenience wrapper for the single-stream case.
    ///
    /// For multiple concurrent streams sharing one loaded model, use
    /// [`NemotronHandle::load`] + [`Nemotron::from_shared`] instead.
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        Ok(Self::from_shared(&NemotronHandle::load(path, exec_config)?))
    }

    /// Spawn a new Nemotron instance bound to a shared model.
    ///
    /// Each instance owns independent decoder state (~7.5 MB) while the
    /// expensive ONNX session is shared through the handle.
    /// The model lock is held only during encoder/decoder inference
    /// (~20-50 ms per 560 ms audio chunk).
    ///
    /// For the multilingual variant the new instance defaults to `auto`
    /// (prompt index 101) — the model picks the language itself. Override
    /// via [`Self::set_target_lang`] when you know the language; that's
    /// strictly more accurate.
    pub fn from_shared(handle: &NemotronHandle) -> Self {
        let encoder_cache = NemotronEncoderCache::with_dims(
            handle.num_encoder_layers,
            handle.left_context,
            handle.hidden_dim,
            handle.conv_context,
        );

        let prompt_index = match handle.mode {
            NemotronMode::Multilingual => Some(101),
            NemotronMode::EnglishOnly => None,
        };

        Self {
            model: Arc::clone(&handle.model),
            vocab: Arc::clone(&handle.vocab),
            mel_basis: Arc::clone(&handle.mel_basis),
            mode: handle.mode,
            num_encoder_layers: handle.num_encoder_layers,
            hidden_dim: handle.hidden_dim,
            left_context: handle.left_context,
            conv_context: handle.conv_context,
            vocab_size: handle.vocab_size,
            blank_id: handle.blank_id,
            lang_tag_ids: Arc::clone(&handle.lang_tag_ids),
            encoder_cache,
            state_1: Array3::zeros((handle.decoder_lstm_layers, 1, handle.decoder_lstm_dim)),
            state_2: Array3::zeros((handle.decoder_lstm_layers, 1, handle.decoder_lstm_dim)),
            last_token: handle.blank_id as i32,
            prompt_index,
            audio_buffer: Vec::new(),
            audio_processed: 0,
            chunk_idx: 0,
            accumulated_tokens: Vec::new(),
        }
    }

    /// Which variant this instance wraps.
    pub fn mode(&self) -> NemotronMode {
        self.mode
    }

    /// Set the target language for the multilingual model. Accepts any key
    /// from [`NemotronHandle::available_languages`] (e.g. `"en-US"`, `"es-ES"`,
    /// `"ja-JP"`, `"auto"` for language-agnostic decoding).
    ///
    /// **Quality note:** NVIDIA's model card documents 40 language-locales
    /// across 3 tiers (transcription-ready, broad-coverage, adaptation-ready).
    /// Adaptation-ready locales need fine-tuning for production quality.
    /// The full prompt dictionary accepts additional codes (e.g. `qu-PE`,
    /// `mi-NZ`, `haw-US`) that the model has prompt slots for but are not
    /// in the model card — those will run, but accuracy is not guaranteed.
    /// See: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
    ///
    /// Returns an error on the English-only variant or for an unknown language.
    /// The new language takes effect on the next encoder call — for clean
    /// switching mid-utterance you usually also want [`Self::reset`].
    pub fn set_target_lang(&mut self, lang: &str) -> Result<()> {
        if self.mode != NemotronMode::Multilingual {
            return Err(Error::Config(
                "set_target_lang is only available on the multilingual variant".into(),
            ));
        }
        let idx = PROMPT_DICTIONARY
            .iter()
            .find_map(|(k, v)| (*k == lang).then_some(*v))
            .ok_or_else(|| {
                Error::Config(format!(
                    "Unknown target language '{lang}'. Try one of: en-US, es-ES, de-DE, fr-FR, ja-JP, zh-CN, auto, ..."
                ))
            })?;
        self.prompt_index = Some(idx);
        Ok(())
    }

    /// Reset all state for new utterance. Preserves the configured target
    /// language (call [`Self::set_target_lang`] again to change it).
    pub fn reset(&mut self) {
        self.encoder_cache = NemotronEncoderCache::with_dims(
            self.num_encoder_layers,
            self.left_context,
            self.hidden_dim,
            self.conv_context,
        );
        self.state_1.fill(0.0);
        self.state_2.fill(0.0);
        self.last_token = self.blank_id as i32;
        self.audio_buffer.clear();
        self.audio_processed = 0;
        self.chunk_idx = 0;
        self.accumulated_tokens.clear();
    }

    /// Get the full accumulated transcript. Language tag tokens (e.g. `<en-US>`)
    /// emitted by the multilingual model are stripped.
    pub fn get_transcript(&self) -> String {
        let valid: Vec<usize> = self
            .accumulated_tokens
            .iter()
            .copied()
            .filter(|t| *t < self.vocab_size && !self.lang_tag_ids.contains(t))
            .collect();
        self.vocab.decode(&valid)
    }

    /// note that, offline transcription for testing/debugging and for some curious ppl :-). with following function too (transcribe_audio)
    pub fn transcribe_file<P: AsRef<Path>>(&mut self, audio_path: P) -> Result<String> {
        let audio = Self::load_mono_file(audio_path)?;
        self.transcribe_audio(&audio)
    }

    /// Transcribe an audio file (non-streaming) with token-level timestamps.
    /// See [`Self::transcribe_audio_with_timestamps`] for the `mode` argument.
    pub fn transcribe_file_with_timestamps<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        let audio = Self::load_mono_file(audio_path)?;
        self.transcribe_audio_with_timestamps(&audio, mode)
    }

    /// Decode an audio file to mono f32 samples (averaging channels if needed).
    fn load_mono_file<P: AsRef<Path>>(audio_path: P) -> Result<Vec<f32>> {
        let (audio, spec) = crate::audio::load_audio(audio_path)?;
        Ok(if spec.channels > 1 {
            audio
                .chunks(spec.channels as usize)
                .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
                .collect()
        } else {
            audio
        })
    }

    /// Transcribe audio samples (non-streaming)
    pub fn transcribe_audio(&mut self, audio: &[f32]) -> Result<String> {
        let tokens = self.process_audio(audio)?;
        let valid: Vec<usize> = tokens
            .iter()
            .map(|t| t.id)
            .filter(|t| *t < self.vocab_size && !self.lang_tag_ids.contains(t))
            .collect();
        Ok(self.vocab.decode(&valid))
    }

    /// Offline companion to [`Self::transcribe_audio`] returning per-token
    /// information (id, decoded text, log-probability) instead of a joined
    /// string. Every emitted token is included; language-tag tokens (e.g.
    /// `<en-US>`) are NOT stripped, so callers can inspect or filter them.
    pub fn transcribe_audio_with_tokens(&mut self, audio: &[f32]) -> Result<Vec<TokenInfo>> {
        self.process_audio(audio)
    }

    /// Transcribe audio samples (non-streaming) with token-level timestamps.
    ///
    /// `mode` controls whether the returned `tokens` are raw model tokens,
    /// grouped into words, or grouped into sentences (defaults to
    /// [`TimestampMode::Tokens`]). Timing comes from the encoder frame each token
    /// was emitted at (one frame is 80ms), so it is frame-accurate rather than
    /// interpolated. Language-tag and out-of-vocabulary tokens are filtered out.
    pub fn transcribe_audio_with_timestamps(
        &mut self,
        audio: &[f32],
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        let frame_seconds = (SUBSAMPLING_FACTOR * HOP_LENGTH) as f32 / SAMPLE_RATE as f32;
        let raw_tokens: Vec<TimedToken> = self
            .process_audio(audio)?
            .into_iter()
            .filter(|t| t.id < self.vocab_size && !self.lang_tag_ids.contains(&t.id))
            .map(|t| {
                let start = t.local_frame as f32 * frame_seconds;
                TimedToken {
                    text: t.text,
                    start,
                    end: start + frame_seconds,
                }
            })
            .collect();

        let mode = mode.unwrap_or(TimestampMode::Tokens);
        let tokens = process_timestamps(&raw_tokens, mode);
        let text = rebuild_text(&tokens, mode);
        Ok(TranscriptionResult { text, tokens })
    }

    /// Shared offline decode: runs the full chunk loop and returns every
    /// emitted token. `transcribe_audio` filters + joins these into a string.
    fn process_audio(&mut self, audio: &[f32]) -> Result<Vec<TokenInfo>> {
        self.reset();

        let mel = self.compute_mel_spectrogram(audio)?;
        let total_frames = mel.shape()[1];

        if total_frames == 0 {
            return Ok(Vec::new());
        }

        let mut all_tokens: Vec<TokenInfo> = Vec::new();
        let mut buffer_idx = 0;
        let mut chunk_idx = 0;
        // Running count of encoder frames consumed by previous chunks. Added to
        // each token's chunk-local frame so `local_frame` becomes a global index.
        let mut encoder_frame_base = 0usize;

        while buffer_idx < total_frames {
            let chunk_end = (buffer_idx + CHUNK_SIZE).min(total_frames);
            let main_len = chunk_end - buffer_idx;

            let expected_size = PRE_ENCODE_CACHE + CHUNK_SIZE;
            let mut chunk_data = vec![0.0f32; N_MELS * expected_size];

            // Fill pre-encode cache from previous frames
            if chunk_idx > 0 && buffer_idx >= PRE_ENCODE_CACHE {
                let cache_start = buffer_idx - PRE_ENCODE_CACHE;
                for f in 0..PRE_ENCODE_CACHE {
                    for m in 0..N_MELS {
                        chunk_data[m * expected_size + f] = mel[[m, cache_start + f]];
                    }
                }
            }

            // Fill main chunk
            for f in 0..main_len {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + PRE_ENCODE_CACHE + f] = mel[[m, buffer_idx + f]];
                }
            }

            let mel_chunk = Array3::from_shape_vec((1, N_MELS, expected_size), chunk_data)
                .map_err(|e| Error::Model(format!("Failed to create mel chunk: {e}")))?;

            let chunk_length = PRE_ENCODE_CACHE + main_len;

            let (encoded, enc_len, new_cache) = {
                let mut model = self.model.lock().map_err(|e| {
                    Error::Model(format!("Failed to acquire model lock: {e}"))
                })?;
                model.run_encoder(
                    &mel_chunk,
                    chunk_length as i64,
                    &self.encoder_cache,
                    self.prompt_index,
                )?
            };
            self.encoder_cache = new_cache;

            let mut new_tokens = self.decode_chunk_tokens(&encoded, enc_len as usize)?;
            // Rebase chunk-local frame indices onto the global timeline.
            for token in &mut new_tokens {
                token.local_frame += encoder_frame_base;
            }
            all_tokens.extend(new_tokens);

            encoder_frame_base += enc_len as usize;
            buffer_idx += CHUNK_SIZE;
            chunk_idx += 1;
        }

        Ok(all_tokens)
    }

    /// Stream transcribe a chunk of audio (call repeatedly for real-time).
    ///
    /// This buffers raw audio and computes mel spectrograms over the full buffer
    /// to avoid edge effects at chunk boundaries.
    pub fn transcribe_chunk(&mut self, audio_chunk: &[f32]) -> Result<String> {
        let tokens = self.process_chunk(audio_chunk)?;
        let mut result = String::new();
        for t in &tokens {
            if t.id < self.vocab_size && !self.lang_tag_ids.contains(&t.id) {
                result.push_str(&t.text);
            }
        }
        Ok(result)
    }

    /// Streaming companion to [`Self::transcribe_chunk`] returning per-token
    /// information (id, decoded text, log-probability) instead of a joined
    /// string. Every emitted token is included; language-tag tokens (e.g.
    /// `<en-US>`) are NOT stripped, so callers can inspect or filter them.
    pub fn transcribe_chunk_with_tokens(
        &mut self,
        audio_chunk: &[f32],
    ) -> Result<Vec<TokenInfo>> {
        self.process_chunk(audio_chunk)
    }

    /// Shared streaming decode: buffers audio, runs the encoder/decoder for any
    /// ready chunk, accumulates token ids, and returns this call's emitted
    /// tokens. `transcribe_chunk` filters + joins these into a string.
    fn process_chunk(&mut self, audio_chunk: &[f32]) -> Result<Vec<TokenInfo>> {
        // Append raw audio to buffer
        self.audio_buffer.extend_from_slice(audio_chunk);

        // Calculate how many mel frames we can produce from buffered audio
        // mel_frames = 1 + (audio_len + 2*pad - win_length) / hop_length
        // For center=true padding, we need at least win_length samples to get 1 frame
        let total_audio = self.audio_buffer.len();
        if total_audio < WIN_LENGTH {
            return Ok(Vec::new());
        }

        // Compute mel spectrogram over the ENTIRE audio buffer
        let full_mel = self.compute_mel_spectrogram(&self.audio_buffer)?;
        let total_mel_frames = full_mel.shape()[1];

        // Calculate how many mel frames correspond to processed audio
        // Each CHUNK_SIZE mel frames = CHUNK_SIZE * HOP_LENGTH audio samples
        let processed_mel_frames = self.audio_processed / HOP_LENGTH;

        // Check if we have enough NEW frames to process a chunk
        let available_new_frames = total_mel_frames.saturating_sub(processed_mel_frames);
        if available_new_frames < CHUNK_SIZE {
            return Ok(Vec::new());
        }

        // Build encoder input chunk
        let expected_size = PRE_ENCODE_CACHE + CHUNK_SIZE;
        let mut chunk_data = vec![0.0f32; N_MELS * expected_size];

        // Determine the mel frame range for this chunk
        let is_first_chunk = self.chunk_idx == 0;
        let main_start = processed_mel_frames;
        let _main_end = main_start + CHUNK_SIZE;

        if is_first_chunk {
            // First chunk: zero-pad for pre-encode cache
            for f in 0..CHUNK_SIZE.min(total_mel_frames) {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + PRE_ENCODE_CACHE + f] = full_mel[[m, f]];
                }
            }
        } else {
            // Subsequent chunks: include pre-encode cache from previous frames
            let cache_start = main_start.saturating_sub(PRE_ENCODE_CACHE);
            let cache_frames = main_start - cache_start;
            let cache_offset = PRE_ENCODE_CACHE - cache_frames;

            // Fill pre-encode cache
            for f in 0..cache_frames {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + cache_offset + f] =
                        full_mel[[m, cache_start + f]];
                }
            }

            // Fill main chunk
            for f in 0..CHUNK_SIZE.min(total_mel_frames - main_start) {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + PRE_ENCODE_CACHE + f] =
                        full_mel[[m, main_start + f]];
                }
            }
        }

        let mel_chunk = Array3::from_shape_vec((1, N_MELS, expected_size), chunk_data)
            .map_err(|e| Error::Model(format!("Failed to create mel chunk: {e}")))?;

        let (encoded, enc_len, new_cache) = {
            let mut model = self.model.lock().map_err(|e| {
                Error::Model(format!("Failed to acquire model lock: {e}"))
            })?;
            model.run_encoder(
                &mel_chunk,
                expected_size as i64,
                &self.encoder_cache,
                self.prompt_index,
            )?
        };
        self.encoder_cache = new_cache;

        let tokens = self.decode_chunk_tokens(&encoded, enc_len as usize)?;
        // Store all emitted ids unfiltered; `get_transcript` strips lang tags
        // and out-of-vocab ids at read time.
        self.accumulated_tokens.extend(tokens.iter().map(|t| t.id));

        // Advance processed position
        self.audio_processed += CHUNK_SIZE * HOP_LENGTH;
        self.chunk_idx += 1;

        // Trim audio buffer to keep memory bounded
        // Keep enough for pre-encode cache context
        let keep_samples = (PRE_ENCODE_CACHE + CHUNK_SIZE) * HOP_LENGTH + WIN_LENGTH;
        if self.audio_buffer.len() > keep_samples * 2 {
            let remove = self.audio_buffer.len() - keep_samples;
            // Adjust processed counter since we're removing from the start
            let actual_remove = remove.min(self.audio_processed);
            self.audio_buffer.drain(0..actual_remove);
            self.audio_processed -= actual_remove;
        }

        Ok(tokens)
    }

    fn decode_chunk_tokens(
        &mut self,
        encoder_out: &Array3<f32>,
        enc_frames: usize,
    ) -> Result<Vec<TokenInfo>> {
        let mut tokens = Vec::new();
        let hidden_dim = encoder_out.shape()[1];
        let max_symbols_per_step = 10;

        // Lock the model once for the entire decode loop to minimise
        // lock acquire/release overhead (many decoder steps per chunk).
        let mut model = self.model.lock().map_err(|e| {
            Error::Model(format!("Failed to acquire model lock: {e}"))
        })?;

        for t in 0..enc_frames {
            let frame = encoder_out.slice(s![0, .., t]).to_owned();
            let frame = frame
                .to_shape((1, hidden_dim, 1))
                .map_err(|e| Error::Model(format!("Failed to reshape frame: {e}")))?
                .to_owned();

            for _ in 0..max_symbols_per_step {
                let (logits, new_state_1, new_state_2) = model.run_decoder(
                    &frame,
                    self.last_token,
                    &self.state_1,
                    &self.state_2,
                )?;

                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &v) in logits.iter().enumerate() {
                    if v > max_val {
                        max_val = v;
                        max_idx = i;
                    }
                }

                if max_idx == self.blank_id {
                    break;
                }

                let logits_slice = logits
                    .as_slice()
                    .ok_or_else(|| Error::Model("non-contiguous logits".into()))?;
                let logprob = log_softmax_at(logits_slice, max_idx, max_val);
                tokens.push(TokenInfo {
                    id: max_idx,
                    text: self.vocab.decode_single(max_idx),
                    logprob,
                    local_frame: t,
                });
                self.last_token = max_idx as i32;
                self.state_1 = new_state_1;
                self.state_2 = new_state_2;
            }
        }

        Ok(tokens)
    }

    /// Compute log mel spectrogram WITHOUT normalization.
    /// I use capitals because this gave me some trouble on the Python side :(). I realized they dont use it later.
    /// so offc nemo feeding raw log-mel spectrogram values (in decibels) directly to the encoder.
    fn compute_mel_spectrogram(&self, audio: &[f32]) -> Result<Array2<f32>> {
        if audio.is_empty() {
            return Ok(Array2::zeros((N_MELS, 0)));
        }

        let preemph = crate::audio::apply_preemphasis(audio, PREEMPH);
        let spec = crate::audio::stft(&preemph, N_FFT, HOP_LENGTH, WIN_LENGTH)?;
        let mel = self.mel_basis.dot(&spec);

        Ok(mel.mapv(|x| (x + LOG_ZERO_GUARD).ln()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Max logit, as the decode loop computes it before calling `log_softmax_at`.
    fn max_logit(logits: &[f32]) -> f32 {
        logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    #[test]
    fn frame_to_seconds_is_eighty_ms() {
        // Guards against accidental drift in the subsampling/hop/sample-rate
        // constants that feed per-token timestamp timing.
        let frame_seconds = (SUBSAMPLING_FACTOR * HOP_LENGTH) as f32 / SAMPLE_RATE as f32;
        assert!((frame_seconds - 0.08).abs() < 1e-6);
        assert!((7.0 * frame_seconds - 0.56).abs() < 1e-6);
    }

    #[test]
    fn log_softmax_is_finite_and_non_positive() {
        let logits = [2.0f32, -1.0, 0.5, 3.5, -4.0];
        let m = max_logit(&logits);
        for idx in 0..logits.len() {
            let lp = log_softmax_at(&logits, idx, m);
            assert!(lp.is_finite(), "logprob at {idx} must be finite");
            assert!(lp <= 0.0, "log-probability must be <= 0, got {lp}");
        }
    }

    #[test]
    fn log_softmax_peaks_at_argmax() {
        let logits = [2.0f32, -1.0, 0.5, 3.5, -4.0];
        let m = max_logit(&logits);
        let argmax = 3; // 3.5 is the largest
        let best = log_softmax_at(&logits, argmax, m);
        for idx in 0..logits.len() {
            if idx != argmax {
                assert!(
                    log_softmax_at(&logits, idx, m) <= best,
                    "no token may exceed the argmax log-probability"
                );
            }
        }
    }

    #[test]
    fn log_softmax_sums_to_one() {
        let logits = [2.0f32, -1.0, 0.5, 3.5, -4.0];
        let m = max_logit(&logits);
        let total: f32 = (0..logits.len())
            .map(|idx| log_softmax_at(&logits, idx, m).exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-5, "probabilities must sum to 1, got {total}");
    }
}
