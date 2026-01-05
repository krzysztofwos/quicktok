mod tokenizer;

use pyo3::prelude::*;

pub use crate::tokenizer::{basic::BasicTokenizer, Tokenizer};

#[pymodule]
fn quicktok(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BasicTokenizer>()?;
    Ok(())
}
