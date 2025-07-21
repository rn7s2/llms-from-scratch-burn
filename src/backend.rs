use burn::backend::Autodiff;

#[cfg(not(feature = "ndarray"))]
mod backend {
    use super::*;

    pub type Backend = Autodiff<burn::backend::Wgpu>;
    pub type InferenceBackend = burn::backend::Wgpu;
}

#[cfg(feature = "ndarray")]
mod backend {
    use super::*;

    pub type Backend = Autodiff<burn::backend::NdArray>;
    pub type InferenceBackend = burn::backend::NdArray;
}

pub use backend::*;
