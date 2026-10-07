use std::fmt;
#[cfg(feature = "ort")]
use std::path::{Path, PathBuf};
#[cfg(feature = "ort")]
use std::sync::Arc;

#[cfg(feature = "ort")]
use crate::error::Result;
#[cfg(feature = "ort")]
use ort::session::builder::SessionBuilder;
#[cfg(feature = "ort")]
use ort::session::Session;

// Hardware acceleration options. CPU is default and most reliable.
// GPU providers (CUDA, TensorRT, MIGraphX) offer 5-10x speedup but require specific hardware.
// All GPU providers automatically fall back to CPU if they fail.
//
// Note: CoreML EP currently runs slower than CPU for Sortformer/Parakeet models because
// the ONNX graphs have dynamic input shapes, preventing CoreML from building optimised
// execution plans for ANE/GPU. CoreML claims nodes but runs them on CPU with overhead.
//
// WebGPU is experimental and may produce incorrect results.
//
// The Burn* providers run on the pure-Rust burn backend instead of ONNX Runtime.
// They read the same (fp32) .onnx files; every model except Cohere runs on them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    /// ONNX Runtime on the CPU.
    #[cfg(feature = "ort")]
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "tensorrt")]
    TensorRT,
    #[cfg(feature = "coreml")]
    CoreML,
    #[cfg(feature = "directml")]
    DirectML,
    #[cfg(feature = "migraphx")]
    MIGraphX,
    #[cfg(feature = "openvino")]
    OpenVINO,
    #[cfg(feature = "webgpu")]
    WebGPU,
    #[cfg(feature = "nnapi")]
    NNAPI,
    /// burn on the CPU (no ONNX Runtime). Not for Cohere. Uses all cores
    /// (`RAYON_NUM_THREADS` limits them); `intra_threads` applies to ONNX Runtime only.
    #[cfg(feature = "burn")]
    #[cfg_attr(not(feature = "ort"), default)]
    BurnCpu,
    /// burn on the GPU through wgpu (Metal, Vulkan, DX12). Not for Cohere.
    /// Build with `metal` on Apple GPUs. The first run on a machine compiles and tunes kernels
    /// (seconds); they are cached on disk.
    #[cfg(feature = "wgpu")]
    BurnWgpu,
    /// burn's CUDA backend on NVIDIA GPU 0. Needs the NVIDIA driver and CUDA.
    #[cfg(feature = "burn-cuda")]
    BurnCuda,
    /// burn's ROCm backend on AMD GPU 0. Needs ROCm.
    #[cfg(feature = "burn-rocm")]
    BurnRocm,
}

impl ExecutionProvider {
    /// Whether this provider runs on the burn backend rather than ONNX Runtime.
    pub fn is_burn(self) -> bool {
        match self {
            #[cfg(feature = "burn")]
            ExecutionProvider::BurnCpu => true,
            #[cfg(feature = "wgpu")]
            ExecutionProvider::BurnWgpu => true,
            #[cfg(feature = "burn-cuda")]
            ExecutionProvider::BurnCuda => true,
            #[cfg(feature = "burn-rocm")]
            ExecutionProvider::BurnRocm => true,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// Where a model's decoder/joint runs when its encoder runs on `self`. It runs once per token
    /// on tiny tensors, so burn GPU providers hand it to burn's CPU backend.
    pub(crate) fn per_token_provider(self) -> Self {
        match self {
            #[cfg(feature = "wgpu")]
            ExecutionProvider::BurnWgpu => ExecutionProvider::BurnCpu,
            #[cfg(feature = "burn-cuda")]
            ExecutionProvider::BurnCuda => ExecutionProvider::BurnCpu,
            #[cfg(feature = "burn-rocm")]
            ExecutionProvider::BurnRocm => ExecutionProvider::BurnCpu,
            other => other,
        }
    }
}

/// Which compute units the CoreML execution provider may use.
#[cfg(feature = "ort")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoreMLComputeUnits {
    All,
    CpuAndNeuralEngine,
    #[default]
    CpuAndGpu,
    CpuOnly,
}

#[derive(Clone)]
pub struct ModelConfig {
    pub execution_provider: ExecutionProvider,
    pub intra_threads: usize,
    pub inter_threads: usize,
    #[cfg(feature = "ort")]
    pub configure: Option<Arc<dyn Fn(SessionBuilder) -> ort::Result<SessionBuilder> + Send + Sync>>,
    /// Optional cache directory for compiled CoreML models. When set, avoids
    /// recompiling the ONNX-to-CoreML conversion on each session load (~5s).
    /// Only used when execution_provider is CoreML.
    #[cfg(feature = "ort")]
    pub coreml_cache_dir: Option<PathBuf>,
    #[cfg(feature = "ort")]
    pub coreml_compute_units: CoreMLComputeUnits,
}

impl fmt::Debug for ModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("ModelConfig");
        s.field("execution_provider", &self.execution_provider)
            .field("intra_threads", &self.intra_threads)
            .field("inter_threads", &self.inter_threads);
        #[cfg(feature = "ort")]
        s.field(
            "configure",
            &if self.configure.is_some() {
                "<fn>"
            } else {
                "None"
            },
        )
        .field("coreml_cache_dir", &self.coreml_cache_dir)
        .field("coreml_compute_units", &self.coreml_compute_units);
        s.finish()
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::default(),
            intra_threads: 4,
            inter_threads: 1,
            #[cfg(feature = "ort")]
            configure: None,
            #[cfg(feature = "ort")]
            coreml_cache_dir: None,
            #[cfg(feature = "ort")]
            coreml_compute_units: CoreMLComputeUnits::default(),
        }
    }
}

impl ModelConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.execution_provider = provider;
        self
    }

    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = threads;
        self
    }

    pub fn with_inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = threads;
        self
    }

    #[cfg(feature = "ort")]
    pub fn with_custom_configure(
        mut self,
        configure: impl Fn(SessionBuilder) -> ort::Result<SessionBuilder> + Send + Sync + 'static,
    ) -> Self {
        self.configure = Some(Arc::new(configure));
        self
    }

    /// Set cache directory for compiled CoreML models.
    /// Avoids ~5s recompilation on each session load.
    #[cfg(feature = "ort")]
    pub fn with_coreml_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.coreml_cache_dir = Some(path.into());
        self
    }

    /// Select which compute units the CoreML provider may use.
    /// Defaults to [`CoreMLComputeUnits::CpuAndGpu`];
    #[cfg(feature = "ort")]
    pub fn with_coreml_compute_units(mut self, units: CoreMLComputeUnits) -> Self {
        self.coreml_compute_units = units;
        self
    }
    /// Build a session for `path` under this configuration.
    #[cfg(feature = "ort")]
    pub fn build_session(&self, path: &Path) -> Result<Session> {
        let builder = Session::builder()?;
        let mut builder = self.apply_to_session_builder(builder)?;
        Ok(builder.commit_from_file(path)?)
    }

    #[cfg(feature = "ort")]
    pub(crate) fn apply_to_session_builder(
        &self,
        builder: SessionBuilder,
    ) -> Result<SessionBuilder> {
        #[cfg(any(
            feature = "cuda",
            feature = "tensorrt",
            feature = "coreml",
            feature = "directml",
            feature = "migraphx",
            feature = "openvino",
            feature = "webgpu",
            feature = "nnapi"
        ))]
        use ort::ep::CPU as CPUExecutionProvider;
        use ort::session::builder::GraphOptimizationLevel;

        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(self.intra_threads)?
            .with_inter_threads(self.inter_threads)?;

        builder = match self.execution_provider {
            ExecutionProvider::Cpu => builder,

            #[cfg(feature = "cuda")]
            ExecutionProvider::Cuda => builder.with_execution_providers([
                ort::ep::CUDA::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "tensorrt")]
            ExecutionProvider::TensorRT => builder.with_execution_providers([
                ort::ep::TensorRT::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "coreml")]
            ExecutionProvider::CoreML => {
                use ort::ep::coreml::{ComputeUnits, CoreML};
                let units = match self.coreml_compute_units {
                    CoreMLComputeUnits::All => ComputeUnits::All,
                    CoreMLComputeUnits::CpuAndNeuralEngine => ComputeUnits::CPUAndNeuralEngine,
                    CoreMLComputeUnits::CpuAndGpu => ComputeUnits::CPUAndGPU,
                    CoreMLComputeUnits::CpuOnly => ComputeUnits::CPUOnly,
                };
                let mut coreml = CoreML::default().with_compute_units(units);

                if let Some(cache_dir) = &self.coreml_cache_dir {
                    coreml = coreml.with_model_cache_dir(cache_dir.to_string_lossy());
                }

                builder.with_execution_providers([
                    coreml.build(),
                    CPUExecutionProvider::default().build().error_on_failure(),
                ])?
            }

            #[cfg(feature = "directml")]
            ExecutionProvider::DirectML => builder.with_execution_providers([
                ort::ep::DirectML::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "migraphx")]
            ExecutionProvider::MIGraphX => builder.with_execution_providers([
                ort::ep::MIGraphX::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "openvino")]
            ExecutionProvider::OpenVINO => builder.with_execution_providers([
                ort::ep::OpenVINO::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "webgpu")]
            ExecutionProvider::WebGPU => builder.with_execution_providers([
                ort::ep::WebGPU::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "nnapi")]
            ExecutionProvider::NNAPI => builder.with_execution_providers([
                ort::ep::NNAPI::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[allow(unreachable_patterns)]
            provider => {
                return Err(crate::error::Error::Config(format!(
                    "{provider:?} runs on the burn backend, not ONNX Runtime; this model supports only ONNX Runtime providers"
                )));
            }
        };

        if let Some(configure) = self.configure.as_ref() {
            builder = configure(builder)?;
        }

        Ok(builder)
    }
}
