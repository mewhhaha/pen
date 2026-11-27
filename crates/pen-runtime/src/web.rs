//! The optional startup files present in a static web export.

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct WebFiles {
    pub scene: bool,
    pub devtools: bool,
    pub saved: bool,
    pub captions: bool,
    pub locales: bool,
}
