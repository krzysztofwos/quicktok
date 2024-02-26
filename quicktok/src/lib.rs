mod tokenizer;

use pyo3::prelude::*;

pub use crate::tokenizer::{basic::BasicTokenizer, Tokenizer};

#[pymodule]
fn quicktok(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<BasicTokenizer>()?;
    Ok(())
}
