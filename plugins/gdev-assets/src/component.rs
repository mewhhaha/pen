//! The component: wit/pen-plugin.wit's `assets` over a [`crate::Generator`],
//! which this instance keeps between pen's calls, reading the checkout
//! through pen's `files`.
use std::cell::RefCell;

wit_bindgen::generate!({ path: "../../wit", world: "assets-plugin" });

use exports::pen::plugin::assets::{self, Generated};
use pen::plugin::{files, types};

struct Plugin;

export!(Plugin);

thread_local! {
    static GENERATOR: RefCell<crate::Generator> = RefCell::default();
}

/// The checkout's files, as pen grants them.
struct Host;

impl crate::Files for Host {
    fn read(&self, path: &str) -> Result<Vec<u8>, String> {
        files::read(path)
    }

    fn read_directory(&self, directory: &str) -> Result<Vec<String>, String> {
        files::read_directory(directory)
    }

    fn exists(&self, path: &str) -> bool {
        files::exists(path)
    }
}

impl assets::Guest for Plugin {
    fn generate(program: types::Program) -> Result<Generated, String> {
        let program = crate::Program { root: program.root, entry: program.entry, settings: program.settings };
        let generated = GENERATOR
            .with_borrow_mut(|generator| generator.generate(&program, &Host))
            .map_err(|error| error.to_string())?;
        Ok(Generated {
            modules: generated
                .modules
                .into_iter()
                .map(|module| types::Overlay { path: module.path, contents: Some(module.source) })
                .collect(),
            dependencies: generated.dependencies,
        })
    }
}
