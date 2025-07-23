use burn::backend::Autodiff;

#[cfg(not(feature = "ndarray"))]
mod backend {
    use super::*;

    pub type TrainBackend = Autodiff<burn::backend::Wgpu>;
    pub type Backend = burn::backend::Wgpu;
}

#[cfg(feature = "ndarray")]
mod backend {
    use super::*;

    pub type TrainBackend = Autodiff<burn::backend::NdArray>;
    pub type Backend = burn::backend::NdArray;
}

pub use backend::*;
