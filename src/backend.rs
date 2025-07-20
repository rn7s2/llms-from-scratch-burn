use burn::backend::Autodiff;

#[cfg(not(feature = "ndarray"))]
pub type Backend = Autodiff<burn::backend::Wgpu>;

#[cfg(feature = "ndarray")]
pub type Backend = Autodiff<burn::backend::NdArray>;
