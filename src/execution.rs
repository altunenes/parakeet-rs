use crate::error::Result;
use ort::session::builder::SessionBuilder;

// Hardware acceleration options. CPU is default and most reliable.
// GPU providers (CUDA, TensorRT, ROCm) offer 5-10x speedup but require specific hardware.
//
// CoreML (macOS) currently fails with this model:
// Error: "Unable to compute the prediction using a neural network model...broken/unsupported model"
// The Parakeet ONNX contains operations CoreML doesn't support (likely the attention mechanism).
// This needs to be fixed in a future version - for now just use CPU or CUDA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
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
    #[cfg(feature = "rocm")]
    ROCm,
    #[cfg(feature = "openvino")]
    OpenVINO,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub execution_provider: ExecutionProvider,
    pub intra_threads: usize,
    pub inter_threads: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::default(),
            intra_threads: 4,
            inter_threads: 1,
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

    pub(crate) fn apply_to_session_builder(
        &self,
        builder: SessionBuilder,
    ) -> Result<SessionBuilder> {
        use ort::session::builder::GraphOptimizationLevel;

        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(self.intra_threads)?
            .with_inter_threads(self.inter_threads)?;

        builder = match self.execution_provider {
            ExecutionProvider::Cpu => builder,

            #[cfg(feature = "cuda")]
            ExecutionProvider::Cuda => builder.with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
            ])?,

            #[cfg(feature = "tensorrt")]
            ExecutionProvider::TensorRT => builder.with_execution_providers([
                ort::execution_providers::TensorRTExecutionProvider::default().build(),
            ])?,

            #[cfg(feature = "coreml")]
            ExecutionProvider::CoreML => builder.with_execution_providers([
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ])?,

            #[cfg(feature = "directml")]
            ExecutionProvider::DirectML => builder.with_execution_providers([
                ort::execution_providers::DirectMLExecutionProvider::default().build(),
            ])?,

            #[cfg(feature = "rocm")]
            ExecutionProvider::ROCm => builder.with_execution_providers([
                ort::execution_providers::ROCMExecutionProvider::default().build(),
            ])?,

            #[cfg(feature = "openvino")]
            ExecutionProvider::OpenVINO => builder.with_execution_providers([
                ort::execution_providers::OpenVINOExecutionProvider::default().build(),
            ])?,
        };

        Ok(builder)
    }
}
